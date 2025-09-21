# Simple examples

These examples demonstrate very basic parts of the simulator, but were useful during development and kept here as they demonstrate flexibility. You first need to add the root of the repository to your _PYTHONPATH_

```bash
export PYTHONPATH=~/tt-sim:$PYTHONPATH
```

Then from within each example directory execute the code, e.g. _python3 ex1.py_ . These all use assertions to check the results and ensure correctness, so are fairly useful as a basic set of tests too.

* [one](https://github.com/mesham/tt-sim/tree/main/driver/simple/ex1) Sets up different memories and maps them to ranges in a memory space. Then accesses these different addresses ensuring that data is routed to the correct memory space.
