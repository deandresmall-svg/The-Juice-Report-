TypeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/the-juice-report-/mlb_hr_dashboard_sleek_top10_parlays.py", line 4673, in <module>
    + pitcher_summary["Pitcher_Team"].fillna("?")
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/generic.py", line 7372, in fillna
    new_data = self._mgr.fillna(
        value=value, limit=limit, inplace=inplace, downcast=downcast
    )
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/internals/base.py", line 186, in fillna
    return self.apply_with_block(
           ~~~~~~~~~~~~~~~~~~~~~^
        "fillna",
        ^^^^^^^^^
    ...<5 lines>...
        already_warned=_AlreadyWarned(),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/internals/blocks.py", line 2407, in fillna
    new_values = self.values.fillna(value=value, method=None, limit=limit)
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/arrays/_mixins.py", line 376, in fillna
    self._validate_setitem_value(value)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/arrays/categorical.py", line 1590, in _validate_setitem_value
    return self._validate_scalar(value)
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/arrays/categorical.py", line 1615, in _validate_scalar
    raise TypeError(
    ...<2 lines>...
    ) from None
