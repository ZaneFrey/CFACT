from __future__ import annotations

from pathlib import Path

import tools.selection as selection_module
from tools.selection import select_files_by_local_timerange


def test_file_selection_uses_local_timezone(monkeypatch):
    files = [str(Path("C:/fixtures") / f"isfs_cfact_hr_product_20220220_{hour:02d}.nc") for hour in (16, 17, 18, 19, 20)]
    monkeypatch.setattr(selection_module, "normalize_nc_inputs", lambda *args, **kwargs: files)
    selected, table = select_files_by_local_timerange(
        "unused",
        "2022-02-20 10:00:00",
        "2022-02-20 12:00:00",
        "America/Denver",
    )
    names = [Path(path).name for path in selected]
    assert names == [
        "isfs_cfact_hr_product_20220220_17.nc",
        "isfs_cfact_hr_product_20220220_18.nc",
        "isfs_cfact_hr_product_20220220_19.nc",
    ]
    assert table["Selected"].sum() == 3
