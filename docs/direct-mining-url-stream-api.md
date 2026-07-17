# 直接视频 URL 场景挖掘接口文档

## 1. 接口概述

该接口用于对一个可直接下载的在线视频进行场景标签挖掘。调用方传入视频下载地址和 HMAC-SHA256 签名信息，服务端校验通过后下载视频到本地缓存，并以流式 NDJSON 形式返回处理进度和最终挖掘结果。

该接口只返回挖掘结果，不会上传视频到 MinIO，不会写入视频库，也不会写入 Milvus。

## 2. 接口信息

| 项目 | 说明 |
|---|---|
| 接口地址 | `/api/mining/url/stream` |
| 请求方法 | `POST` |
| 请求类型 | `application/json` |
| 响应类型 | `application/x-ndjson; charset=utf-8` |
| 响应方式 | 流式响应，一行一个 JSON 对象 |
| 鉴权方式 | HMAC-SHA256 签名 |

## 3. 请求参数

### 3.1 Body 参数

| 参数名 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `video_url` | string | 是 | 可直接下载的视频 HTTP/HTTPS 地址 |

### 3.2 Header 参数

| Header | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `X-Timestamp` | integer | 是 | 秒级 Unix 时间戳，用于请求过期校验 |
| `X-Nonce` | string | 是 | 随机字符串，用于防止请求重放，最大长度 128 |
| `X-Content-SHA256` | string | 是 | 原始请求体 UTF-8 字节的 SHA256 小写十六进制摘要 |
| `X-Signature` | string | 是 | HMAC-SHA256 签名，小写十六进制字符串 |

### 3.3 请求示例

```json
{
  "video_url": "https://example.com/path/video.mp4"
}
```

## 4. 签名规则

### 4.1 请求体摘要

调用方必须先对实际发送的原始请求体计算 SHA256 摘要：

```text
body_hash = SHA256(raw_body_bytes)
```

示例：

```json
{"video_url":"https://example.com/path/video.mp4"}
```

如果实际发送的请求体是以上 JSON 字符串，则 `body_hash` 必须基于这个字符串的 UTF-8 字节计算。服务端同样基于收到的原始 body 字节计算摘要，不会对 JSON 重新排序或重新序列化。

### 4.2 签名原文

签名原文由以下字段按顺序使用换行符 `\n` 拼接：

```text
HTTP_METHOD + "\n" +
REQUEST_PATH + "\n" +
X-Timestamp + "\n" +
X-Nonce + "\n" +
X-Content-SHA256
```

当前接口的签名原文示例：

```text
POST
/api/mining/url/stream
1784260000
9f5e3b7a5f4a4a2b8d4f9c3a1b2c6d7e
<body_hash>
```

### 4.3 签名算法

```text
sign = HMAC-SHA256(签名原文, 私钥)
```

签名结果使用小写十六进制字符串。

### 4.4 校验规则

| 规则 | 说明 |
|---|---|
| 时间窗口 | `X-Timestamp` 默认必须在服务端当前时间前后 300 秒内 |
| 防重放 | 同一个 `X-Nonce` 在有效时间窗口内只能使用一次 |
| Body 摘要 | 服务端计算原始请求体 SHA256，并与请求头 `X-Content-SHA256` 比对 |
| 签名校验 | 服务端使用本地固定私钥重新计算签名，并与请求头 `X-Signature` 做常量时间比较 |
| 私钥传输 | 私钥不通过接口传输，只保存在调用方和服务端配置中 |

## 5. 响应格式

该接口为流式响应，每一行都是一个独立 JSON 对象，统一使用以下结构：

```json
{
  "code": 0,
  "msg": "success",
  "data": {}
}
```

### 5.1 进度事件

处理中会返回多条进度事件，`data.type` 为 `progress`。

```json
{
  "code": 0,
  "msg": "下载/缓存视频中...",
  "data": {
    "type": "progress",
    "stage": "downloading",
    "detail": {},
    "timestamp": 1784260000123
  }
}
```

进度字段说明：

| 字段 | 类型 | 说明 |
|---|---:|---|
| `code` | integer | 状态码，`0` 表示当前事件正常 |
| `msg` | string | 当前进度描述 |
| `data.type` | string | 固定为 `progress` |
| `data.stage` | string | 当前处理阶段 |
| `data.detail` | object | 阶段补充信息 |
| `data.timestamp` | integer | 毫秒级事件时间戳 |

常见 `stage`：

| stage | 说明 |
|---|---|
| `queued` | 任务已提交 |
| `downloading` | 下载或复用视频缓存 |
| `vlm` | 多模态大模型分析中 |
| `processing` | 通用处理中阶段 |

### 5.2 成功结果事件

挖掘成功时，最后会返回一条 `data.type = result` 的结果事件。

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "type": "result",
    "video_url": "https://example.com/path/video.mp4",
    "tags": ["白天", "晴天", "城市地面道路"],
    "pred": {
      "自然时间段": ["白天"],
      "气象条件": ["晴天"],
      "主干道路级别": ["城市地面道路"]
    },
    "abnormal_event_times": [],
    "timestamp": 1784260009123
  }
}
```

结果字段说明：

| 字段 | 类型 | 说明 |
|---|---:|---|
| `code` | integer | 状态码，成功为 `0` |
| `msg` | string | 成功时为 `success` |
| `data.type` | string | 固定为 `result` |
| `data.video_url` | string | 请求传入的视频地址 |
| `data.tags` | string[] | 去重后的标签列表，最多返回前 10 个 |
| `data.pred` | object | 按标签类别分组的完整预测结果 |
| `data.abnormal_event_times` | object[] | 异常事件时间段列表 |
| `data.timestamp` | integer | 毫秒级事件时间戳 |

### 5.3 错误响应

参数错误或验签失败时，接口会直接返回普通 JSON，不进入流式处理。

```json
{
  "code": 401,
  "msg": "签名错误",
  "data": null
}
```

处理过程中发生错误时，会在 NDJSON 流中返回错误事件。

```json
{
  "code": 500,
  "msg": "挖掘失败: ...",
  "data": {
    "type": "error",
    "timestamp": 1784260009123
  }
}
```

常见错误码：

| code | 说明 |
|---:|---|
| `0` | 成功或进度事件正常 |
| `400` | 请求参数错误，例如缺少字段、URL 协议不支持 |
| `401` | 签名错误、签名过期或 `X-Nonce` 重复 |
| `500` | 服务端配置错误或挖掘过程失败 |

## 6. 调用示例

### 6.1 curl 示例

以下示例使用 `openssl` 生成签名。实际调用时不要在客户端日志中输出私钥。

```bash
API_URL="http://127.0.0.1:30012/api/mining/url/stream"
VIDEO_URL="https://example.com/path/video.mp4"
SECRET="your-secret"
TIMESTAMP="$(date +%s)"
ONCE="$(uuidgen | tr -d '-')"
BODY="{\"video_url\":\"${VIDEO_URL}\"}"
BODY_HASH="$(printf '%s' "$BODY" | openssl dgst -sha256 -hex | awk '{print $2}')"
STRING_TO_SIGN="$(printf 'POST\n/api/mining/url/stream\n%s\n%s\n%s' "$TIMESTAMP" "$ONCE" "$BODY_HASH")"
SIGN="$(printf '%s' "$STRING_TO_SIGN" | openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print $2}')"

curl -N -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/x-ndjson" \
  -H "X-Timestamp: ${TIMESTAMP}" \
  -H "X-Nonce: ${ONCE}" \
  -H "X-Content-SHA256: ${BODY_HASH}" \
  -H "X-Signature: ${SIGN}" \
  -d "$BODY"
```

### 6.2 Java 签名示例

```java
String videoUrl = "https://example.com/path/video.mp4";
String body = "{\"video_url\":\"" + videoUrl + "\"}";
String bodyHash = sha256Hex(body);
long timestamp = System.currentTimeMillis() / 1000L;
String once = UUID.randomUUID().toString().replace("-", "");

String raw = "POST\n"
        + "/api/mining/url/stream\n"
        + timestamp + "\n"
        + once + "\n"
        + bodyHash;
String sign = calculateSign(raw, secretKey).toLowerCase();
```

`sha256Hex` 和 `calculateSign` 示例：

```java
public static String sha256Hex(String content) {
    try {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] hash = digest.digest(content.getBytes(StandardCharsets.UTF_8));
        return Hex.encodeHexString(hash);
    } catch (Exception e) {
        throw new RuntimeException("SHA256摘要计算失败", e);
    }
}

public static String calculateSign(String content, String secretKey) {
    try {
        final String algorithm = "HmacSHA256";
        SecretKeySpec secretKeySpec = new SecretKeySpec(
                secretKey.getBytes(StandardCharsets.UTF_8),
                algorithm
        );
        Mac mac = Mac.getInstance(algorithm);
        mac.init(secretKeySpec);
        byte[] signBytes = mac.doFinal(content.getBytes(StandardCharsets.UTF_8));
        return Hex.encodeHexString(signBytes);
    } catch (Exception e) {
        throw new RuntimeException("签名计算失败", e);
    }
}
```

## 7. 注意事项

1. `video_url` 必须是可直接下载的视频地址，支持 `http` 和 `https`。
2. 接口默认不允许下载内网、回环地址或保留地址，避免 SSRF 风险。
3. `X-Timestamp` 使用秒级 Unix 时间戳，不是毫秒级。
4. `X-Content-SHA256` 必须基于实际发送的原始请求体计算，服务端不会对 JSON 重新排序或重新序列化后再验签。
5. 响应为 NDJSON 流，客户端需要按行解析 JSON。
6. 使用 `curl` 调试时建议加 `-N`，避免客户端缓冲导致看不到实时进度。
7. 该接口不保存视频资产和挖掘结果；如需入库和检索，应使用视频入库处理流程。
