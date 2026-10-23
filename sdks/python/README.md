# memorypilot (Python)

REST client for MemoryPilot server mode. Not published to PyPI.

```python
from memorypilot import MemoryPilot

client = MemoryPilot("https://memory.example", "mpk_…").for_subject("alice")
client.add_memory(content="Alice prefers green tea")
print(client.search(query="tea")["results"])
```
