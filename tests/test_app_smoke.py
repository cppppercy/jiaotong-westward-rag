from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_with_or_without_local_documents() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=10)
    app.run()
    assert not app.exception
    assert app.title[0].value == "交通大学西迁知识库问答"
    if app.warning:
        assert any("知识库为空" in item.value for item in app.warning)
    else:
        assert any("片段数：" in item.value for item in app.caption)
