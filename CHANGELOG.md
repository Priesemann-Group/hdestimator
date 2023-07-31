
### 23-07-31
- built a version based on numba. saves _a lot_ of code, and the compiler dependencies. tests run about 2x faster, the wrapper seems 0.5 slower. room for improvement.
- added dicts as backend to replace hdf5, when persistent is false. this reduces the disk io. basic consistency tests work out.
    - plan for csv: just use pandas.
- for hdf5, storing the symbol_counts now as a 2-column array of ints. strings make trouble.

### 23-07-04
- refactoring:
    - had to rename `src` folder to `hdestimator` to get sane python import statements.
    - removed the `hde_` prefix from filenames in above folder.

### 23-07-04
- working on usage as a python module (in addition to the command line)
    - added python wrapper to hd_api and started required changes:
    - changed imports to `from . import ` so they work on module level
    - added `dtype=object` to ragged arrays when preparing spikes to avoid numpy warning
    - added logging module in some places. more consistent rewrite needed.
    - mid-term plan is to replace the hdf5 file on disk with a dict in memory, to reuse
        most of the existing code.
