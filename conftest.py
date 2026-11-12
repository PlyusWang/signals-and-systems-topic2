"""Keep Qt alive until all GUI tests and worker cleanup have completed."""
import pytest
import pyqtgraph as pg
from PySide6 import QtWidgets


@pytest.fixture(scope="session", autouse=True)
def qt_application():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    previous_rate = pg.getConfigOption("mouseRateLimit")
    pg.setConfigOption("mouseRateLimit", 0)
    yield app
    for window in app.topLevelWidgets():
        window.close()
    app.processEvents()
    pg.setConfigOption("mouseRateLimit", previous_rate)
