# AK2数据采集格式定义

> 文档版本: V1.0  
> 编制日期: 2026-06-04  
> 适用范围: AK2超声波场景数据采集与离线训练入库  
> 当前约束: 现阶段不考虑弹性波数据

---

## 1. 目的与范围

本文件用于统一下一阶段AK2场景数据采集格式，目标是同时满足:

1. 车端原始数据可追溯（面向协议层与问题定位）。
2. 离线训练/仿真可直接消费（面向算法与工具链）。
3. 跨芯片兼容（佑航AK2与极海AK2同一数据模型）。

本规范仅覆盖超声波链路，不包含弹性波采集、标注和建模。

---

## 2. 手册关键信息归纳（用于格式设计）

### 2.1 佑航DJ628.30（AK2）

从产品说明书可提炼的采集相关能力:

1. 总线: ESI/DSI3，速率最高约888kbit/s（手册中同时出现1M/s接口表述与888kb/s兼容表述，工程按888kbit/s预算）。
2. 回波处理: 支持EDI结构化输出（回波类型、时间戳、置信度）。
3. 波形: 支持原始包络输出，可通过DMA搬运到SRAM。
4. 模拟前端: 16bit ADC，适合高分辨率包络采样。

### 2.2 极海G32A217（AK2）

从数据手册可提炼的采集相关能力:

1. 总线: DSI3，最高444kbit/s。
2. 控制核: Cortex-M0，资源较紧（SRAM 4KB），更适合结构化数据+按需波形策略。
3. 算法侧: 支持自动阈值、近场采集、包络/原始数据输出。

### 2.3 统一采集策略结论

为兼容两类AK2芯片并保证量产可落地，建议采用两层数据定义:

1. L0原始采集层: 保留总线报文和芯片原始字段，不做语义丢失。
2. L1标准训练层: 固化为统一帧结构，供算法训练、仿真和回放。

---

## 3. 数据集目录规范

每个采集会话使用一个独立目录，命名如下:

session_YYYYMMDD_NNN

目录结构定义:

```text
TestData/
  index.json
  session_YYYYMMDD_NNN/
    session_meta.json
    can_signals.csv
    esi_frames.bin
    ground_truth.json
    checksums.json
```

字段说明:

1. index.json: 数据集索引（会话列表）。
2. session_meta.json: 会话元数据。
3. can_signals.csv: 车辆状态时序数据。
4. esi_frames.bin: L1标准训练层二进制帧（主文件）。
5. ground_truth.json: 标注结果（类别、事件标签）。
6. checksums.json: 文件校验值（SHA256）。

说明: 若现场仅采原始总线，可先落L0，再离线转换生成esi_frames.bin。

---

## 4. 会话元数据格式（session_meta.json）

### 4.1 JSON字段定义

```json
{
  "session_id": "session_20260604_001",
  "date": "2026-06-04",
  "weather": "Clear",
  "temperature_c": 27.5,
  "vehicle_model": "AK2 Test Vehicle",
  "road_type": "Underground Parking",
  "description": "Front obstacle approach, no collision",
  "chip_vendor": "YOUHANG|GEEHY",
  "chip_model": "DJ628.30|G32A217",
  "bus_type": "ESI|DSI3",
  "bus_bitrate_kbps": 888,
  "sensor_count": 12,
  "frame_rate_hz": 10,
  "envelope_points": 256,
  "elastic_enabled": false,
  "total_frames": 120,
  "timezone": "Asia/Shanghai"
}
```

### 4.2 必填字段

必须包含:

1. session_id
2. chip_vendor
3. chip_model
4. bus_type
5. sensor_count
6. frame_rate_hz
7. envelope_points
8. elastic_enabled
9. total_frames

---

## 5. 车辆信号格式（can_signals.csv）

### 5.1 表头定义

```csv
timestamp_ms,speed_kmh,steering_angle_deg,gear
```

### 5.2 字段约束

1. timestamp_ms: 浮点毫秒，单调递增。
2. speed_kmh: km/h。
3. steering_angle_deg: 方向盘或前轮转角，需在meta中注明定义。
4. gear: 枚举值 P/R/N/D。

### 5.3 采样建议

1. 采样率建议100Hz。
2. 插值到超声波帧时，以最近邻或线性插值，策略写入转换日志。

---

## 6. 超声波主帧格式（esi_frames.bin）

## 6.1 设计原则

1. 固定帧长，便于高速回放与随机seek。
2. 强类型字段，避免训练前重复解析协议。
3. 与当前仿真程序保持兼容。
4. 弹性波阶段关闭，但保留兼容区。

### 6.2 字节序与数据类型

1. 字节序: little-endian。
2. 浮点: IEEE754 float32。
3. 无符号: uint8/uint32。

### 6.3 单帧布局（固定长度）

```text
offset  size      field
0       4         frame_id (uint32)
4       4         timestamp_ms (float32)
8       48        edi_distance[12] (float32)
56      48        edi_amplitude[12] (float32)
104     12        edi_confidence[12] (uint8)
116     12        edi_echo_type[12] (uint8)
128     12288     envelopes[12][256] (float32)
12416   960       elastic_features[12][20] (float32, reserved)

Total frame size = 13376 bytes
```

### 6.4 当前阶段（不考虑弹性波）处理规则

1. elastic_features区必须保留。
2. elastic_features全部填0.0。
3. ground_truth中的碰撞字段固定为无碰撞（如需要）。
4. elastic_enabled固定为false。

这样可确保后续启用弹性波时，文件格式不破坏兼容性。

---

## 7. 字段物理意义与取值约束

### 7.1 EDI字段

1. edi_distance[i]: 第i路测距值，单位m，推荐范围0.0~6.5。
2. edi_amplitude[i]: 第i路回波幅值，归一化到0.0~1.0。
3. edi_confidence[i]: 第i路置信度，0~100（统一映射；若源为0~255，转换时线性缩放）。
4. edi_echo_type[i]: 回波类型枚举。

建议回波类型统一编码:

1. 0: no_echo
2. 1: valid_echo
3. 2: near_field
4. 3: clutter_or_multi_path

### 7.2 包络字段

1. envelopes[i][j]表示第i路传感器第j个采样点。
2. 统一点数为256点。
3. 幅值统一归一化到0.0~1.0。
4. 若源数据点数不为256，使用重采样（线性插值）并记录方法。

---

## 8. L0原始采集层建议（可选但强烈推荐）

为保障可追溯，建议每个会话保留原始总线记录文件:

```text
raw_bus/
  esi_or_dsi3_frames.log
  bus_decode_map.json
```

建议最小字段:

1. host_timestamp_us
2. sensor_id
3. message_type
4. payload_hex
5. crc_ok

说明: L0文件不直接用于训练，但用于回放重建与协议问题定位。

---

## 9. ground_truth.json（超声波阶段）

### 9.1 任务范围

当前仅做超声波场景标签:

1. 12路类别标签（9类）。
2. 无碰撞标记。

### 9.2 结构示例

```json
{
  "session_id": "session_20260604_001",
  "num_classes": 9,
  "class_names": {
    "0": "Wall",
    "1": "Vehicle",
    "2": "Pedestrian",
    "3": "Soft",
    "4": "Open",
    "5": "Clutter",
    "6": "Overhead",
    "7": "Curb",
    "8": "Wet"
  },
  "frames": {
    "0": {
      "class_ids": [4,4,4,4,4,4,0,0,0,0,0,0],
      "has_collision": false,
      "collision_type": 0
    }
  }
}
```

约束:

1. class_ids长度必须等于sensor_count。
2. has_collision固定false（本阶段）。
3. collision_type固定0（本阶段）。

---

## 10. 采集频率、同步与命名要求

### 10.1 频率建议

1. 超声波主帧: 10Hz。
2. CAN信号: 100Hz。
3. 视频参考（可选）: 30fps。

### 10.2 同步要求

1. 所有流统一到同一时间基准。
2. 跨流最大时间误差不超过5ms。
3. 若使用后处理对齐，需输出对齐报告。

### 10.3 传感器编号规范

统一按ch0~ch11存储，映射到安装位S01~S12。

建议映射:

1. ch0~ch5: 前向6路（左到右）。
2. ch6~ch11: 后向6路（左到右）。

---

## 11. 质量校验规则（入库门禁）

每个会话在入库前必须通过以下检查:

1. 文件完整性: 必备文件齐全，checksums.json存在。
2. 帧完整性: esi_frames.bin大小必须为13376的整数倍。
3. 维度一致性: total_frames与实际帧数一致。
4. 时间连续性: timestamp_ms无回退，间隔抖动在允许范围。
5. 值域合法性: 距离、幅值、置信度、类别均在定义区间。
6. 空帧占比: 全零包络帧占比不高于设定阈值（建议<5%）。

---

## 12. 转换与兼容建议（佑航/极海）

### 12.1 厂商差异处理

1. 佑航链路可使用高带宽策略，支持更多包络回传。
2. 极海链路受444kbit/s限制，优先保证EDI稳定回传，包络采用按需回传或降采样。

### 12.2 统一入库原则

无论源协议差异如何，转换后必须满足第6章固定帧格式。

转换步骤建议:

1. 协议解码。
2. 单位与量纲统一。
3. 包络重采样到256点。
4. 置信度归一化到0~100。
5. 写入固定长度帧。
6. 生成转换报告与质量统计。

---

## 13. 最小可执行采集清单

单次会话至少应产出:

1. session_meta.json
2. can_signals.csv
3. esi_frames.bin
4. ground_truth.json
5. checksums.json

满足以上文件即可进入当前AK2仿真与训练流程。

---

## 14. 后续扩展预留

为后续弹性波能力上线，当前格式已预留:

1. elastic_features固定区（960B/帧）。
2. collision相关标注字段。
3. elastic_enabled总开关。

后续只需开放字段赋值与标注策略，不需要变更主文件结构。
