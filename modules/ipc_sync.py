"""
ipc_sync.py  —  Inter-process synchronization module (Windows named pipe)
Implements frame synchronization in dual-instance comparison mode.
Corresponds to design spec section 6.2.

Pipe name: \\\\.\\pipe\\AK2SimSync
Message format (text, newline-terminated):
  FRAME_ADVANCE {frame_id}    Master->Slave: advance to specified frame
  FRAME_ACK {frame_id}        Slave->Master: acknowledge frame complete
  PLAY_STATE {PLAY/PAUSE/STOP}
  SPEED {factor}
  SESSION {path}
"""

import threading
import time
from typing import Callable, Optional

PIPE_NAME = r"\\.\pipe\AK2SimSync"


# ─────────────────────────────────────────────────────────────
# Message type constants
# ─────────────────────────────────────────────────────────────

class MsgType:
    FRAME_ADVANCE = "FRAME_ADVANCE"
    FRAME_ACK     = "FRAME_ACK"
    PLAY_STATE    = "PLAY_STATE"
    SPEED         = "SPEED"
    SESSION       = "SESSION"


# ─────────────────────────────────────────────────────────────
# IPC manager
# ─────────────────────────────────────────────────────────────

class IpcSyncManager:
    """
    Inter-process sync manager for comparison mode.
    role: "master" (process A, starts first, creates pipe server)
          "slave"  (process B, starts second, connects as pipe client)
    """

    def __init__(self, role: str):
        assert role in ("master", "slave"), "role must be 'master' or 'slave'"
        self.role = role
        self._pipe = None
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._callbacks: dict[str, list[Callable]] = {}
        self._pending_ack: Optional[int] = None  # frame_id that Master is waiting to ACK

    # ── Connection management ──────────────────────────────

    def start(self) -> bool:
        """Start the IPC connection; returns True on success.

        For the *master* role ``ConnectNamedPipe`` is moved to a daemon thread
        so the call returns immediately and the GUI stays responsive.
        """
        try:
            import win32pipe, win32file, pywintypes  # type: ignore
        except ImportError:
            print("[IpcSync] WARNING: pywin32 not installed. IPC disabled (single-instance mode).")
            return False

        try:
            if self.role == "master":
                self._pipe = win32pipe.CreateNamedPipe(
                    PIPE_NAME,
                    win32pipe.PIPE_ACCESS_DUPLEX,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                    1, 65536, 65536, 0, None
                )
                # ConnectNamedPipe blocks until a client connects.
                # Run in a daemon thread so the GUI thread is not frozen.
                def _connect_and_recv(pipe_handle):
                    try:
                        win32pipe.ConnectNamedPipe(pipe_handle, None)
                    except Exception as e:
                        print(f"[IpcSync] ConnectNamedPipe error: {e}")
                        return
                    self._running = True
                    self._recv_thread = threading.Thread(
                        target=self._recv_loop, daemon=True)
                    self._recv_thread.start()

                threading.Thread(
                    target=_connect_and_recv,
                    args=(self._pipe,),
                    daemon=True
                ).start()
                return True          # returns immediately; connection happens in background
            else:
                # Slave side: poll until Master creates the pipe
                for _ in range(20):
                    try:
                        self._pipe = win32file.CreateFile(
                            PIPE_NAME,
                            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                            0, None,
                            win32file.OPEN_EXISTING,
                            0, None
                        )
                        break
                    except Exception:
                        time.sleep(0.3)
                else:
                    print("[IpcSync] Timeout waiting for master pipe. Running in single-instance mode.")
                    return False

                self._running = True
                self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
                self._recv_thread.start()
                return True
        except Exception as e:
            print(f"[IpcSync] Failed to start: {e}")
            return False

    def stop(self) -> None:
        self._running = False
        if self._pipe:
            try:
                import win32file
                win32file.CloseHandle(self._pipe)
            except Exception:
                pass
            self._pipe = None

    # ── Message sending ──────────────────────────────

    def send(self, msg_type: str, payload: str = "") -> None:
        """Send one IPC message."""
        if not self._pipe:
            return
        try:
            import win32file
            msg = f"{msg_type} {payload}\n".encode("utf-8")
            win32file.WriteFile(self._pipe, msg)
        except Exception as e:
            print(f"[IpcSync] Send error: {e}")

    def send_frame_advance(self, frame_id: int) -> None:
        """Master: send frame-advance signal and wait for ACK (blocking, up to 200 ms)."""
        self._pending_ack = frame_id
        self.send(MsgType.FRAME_ADVANCE, str(frame_id))
        deadline = time.time() + 0.2
        while self._pending_ack == frame_id and time.time() < deadline:
            time.sleep(0.005)

    def send_ack(self, frame_id: int) -> None:
        """Slave: send frame acknowledgement."""
        self.send(MsgType.FRAME_ACK, str(frame_id))

    def send_play_state(self, state: str) -> None:
        """Send play state (PLAY/PAUSE/STOP)."""
        self.send(MsgType.PLAY_STATE, state)

    def send_speed(self, speed: float) -> None:
        self.send(MsgType.SPEED, str(speed))

    def send_session(self, session_path: str) -> None:
        self.send(MsgType.SESSION, session_path)

    # ── Message receiving ──────────────────────────────

    def on(self, msg_type: str, callback: Callable) -> None:
        """Register a handler callback for the specified message type."""
        self._callbacks.setdefault(msg_type, []).append(callback)

    def _recv_loop(self) -> None:
        """Background thread: continuously receive and dispatch messages."""
        import win32file
        buf = b""
        while self._running:
            try:
                _, data = win32file.ReadFile(self._pipe, 4096)
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._dispatch(line.decode("utf-8").strip())
            except Exception:
                break

    def _dispatch(self, message: str) -> None:
        if not message:
            return
        parts = message.split(" ", 1)
        msg_type = parts[0]
        payload = parts[1] if len(parts) > 1 else ""

        # Handle ACK (consumed internally by Master)
        if msg_type == MsgType.FRAME_ACK:
            try:
                self._pending_ack = None if int(payload) == self._pending_ack else self._pending_ack
            except ValueError:
                pass

        # Dispatch to registered callbacks
        for cb in self._callbacks.get(msg_type, []):
            try:
                cb(payload)
            except Exception as e:
                print(f"[IpcSync] Callback error ({msg_type}): {e}")
