
### 23-07-04
- had to rename `src` folder to `hdestimator` to get sane python import statements.

### 23-07-04
- working on usage as a python module (in addition to the command line)
    - added python wrapper to hd_api and started required changes:
    - changed imports to `from . import ` so they work on module level
    - added `dtype=object` to ragged arrays when preparing spikes to avoid numpy warning
    - added logging module in some places. more consistent rewrite needed.
    - mid-term plan is to replace the hdf5 file on disk with a dict in memory, to reuse
        most of the existing code.
