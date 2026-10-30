# Legacy scripts

`patch_peft*.py` edited the installed `peft` package to work around an
`accelerate`/`peft` version mismatch (missing `clear_device_cache`). With the
current pins (peft >= 0.21, accelerate >= 1.15) the import works without any
patch, so these scripts are no longer needed and should not be run. They are kept
only for reference.
