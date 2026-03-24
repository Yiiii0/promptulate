from unittest import mock

import pytest

import promptulate as pne


def test_init_litellm():
    import litellm

    expected_errors = (litellm.exceptions.APIConnectionError,)
    bad_request_error = getattr(litellm.exceptions, "BadRequestError", None)
    if bad_request_error is not None:
        expected_errors = expected_errors + (bad_request_error,)

    with pytest.raises(expected_errors) as e:
        model = pne.LLMFactory.build(model_name="claude-2")
        model("hello")

    assert "api_key" in str(e.value) or "LLM Provider NOT provided" in str(e.value)


def test_init_zhipu():
    with pytest.raises(KeyError) as e:
        model = pne.LLMFactory.build(model_name="zhipu/glm4")
        model("hello")
        assert (
            str(e.value)
            == "ValueError: ZHIPUAI_API_KEY is not provided. Please set your key."
        )


def test_init_forge(monkeypatch):
    monkeypatch.setenv("FORGE_API_KEY", "forge-key")
    model = pne.LLMFactory.build(model_name="forge/OpenAI/gpt-4o-mini")

    assert model._model == "openai/OpenAI/gpt-4o-mini"
    assert model._model_config["api_key"] == "forge-key"
    assert model._model_config["api_base"] == "https://api.forge.tensorblock.co/v1"


def test_init_forge_allow_custom_base_and_key():
    model = pne.LLMFactory.build(
        model_name="Forge/OpenAI/gpt-4o-mini",
        model_config={
            "api_key": "custom-key",
            "api_base": "https://custom.forge/v1",
            "temperature": 0.1,
        },
    )

    assert model._model == "openai/OpenAI/gpt-4o-mini"
    assert model._model_config["api_key"] == "custom-key"
    assert model._model_config["api_base"] == "https://custom.forge/v1"
    assert model._model_config["temperature"] == 0.1


def test_init_forge_invalid_model_format():
    with pytest.raises(
        ValueError, match="Forge model must use format `forge/Provider/model-name`."
    ):
        pne.LLMFactory.build(model_name="forge/gpt-4o-mini")


@pytest.fixture
def llm_factory():
    return pne.LLMFactory.build("zhipu/glm4")


@pytest.fixture
def mock_response():
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "[start] This is a test [end]"}}]
    }
    return mock_resp


@mock.patch("requests.post")
def test_call(mock_post, llm_factory, mock_response):
    # Use the mock response
    mock_post.return_value = mock_response

    llm_factory.set_private_api_key("my key.hello")
    prompt = """
    Please strictly output the following content.
    ```
    [start] This is a test [end]
    ```
    """
    result = llm_factory(prompt)
    assert result is not None
    assert "[start] This is a test [end]" in result
