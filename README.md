# x.rag

> A simple RAG engine for experiments, no guarantees for
> production performance or stability, meant for rapid iteration
> and prototyping.

## Usage

```python
import rag

ai = rag.AI()
store = rag.Store(ai)

memory_ids = store.add(["memory contents 1", "memory contents 2"])
# child memory (memory_ids[0]) -> parent memory (memory_ids[1])
store.relate(memory_ids[0], memory_ids[1], "memory relationship type")

# returns parent and children memories
store.info(memory_ids[0])

# close the store (very important!)
store.close()
```

