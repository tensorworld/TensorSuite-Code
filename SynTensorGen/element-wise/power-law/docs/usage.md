# Usage

KronWeave has two equivalent entry points:

```bash
kronweave generate --config configs/cp_fastskg_tiny.yaml
```

```python
from kronweave import generate_from_config

result = generate_from_config("configs/cp_fastskg_tiny.yaml")
```

Generated bundles contain:

- `<name>.tns`: TensorSuite text sparse tensor.
- `<name>.tnsb`: TensorSuite binary sparse tensor.
- `<name>_metadata.json`: machine-readable metadata.
- `config.yaml`: copied generation config.
- `README.md`: human-readable bundle summary.
