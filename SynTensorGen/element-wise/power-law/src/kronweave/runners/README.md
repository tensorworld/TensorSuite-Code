# runners

Wrappers around generation backends.

- `__init__.py`: marks this directory as the runners subpackage.
- `fastskg.py`: builds the C++ fastSKG backend when needed, runs it, and reads raw TensorSuite-TNS output back into Python arrays.
