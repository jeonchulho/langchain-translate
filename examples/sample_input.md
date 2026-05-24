# API Migration Guide

Our service endpoint is https://api.example.com/v1/translate and support email is support@example.com.

Please keep identifier names unchanged: user_id, requestPayload, SYSTEM_CONFIG.

| Field | Description | Default |
| --- | --- | --- |
| timeout_ms | Request timeout in milliseconds | 3000 |
| retry_count | Number of retries | 2 |

```python
result = client.translate_text(source_text="Hello", target_language="ko")
print(result)
```
