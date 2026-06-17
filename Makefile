.PHONY: test test-all test-ai test-game test-integration

test:
	env RUN_INTEGRATION= SILICONFLOW_API_KEY= \
		conda run -n chinese_war_game python -m unittest discover -s tests -t . -p "test_*.py" -v

test-all:
	conda run -n chinese_war_game python -m unittest discover -s tests -t . -p "test_*.py" -v

test-ai:
	conda run -n chinese_war_game python -m unittest discover -s tests/ai -t . -p "test_*.py" -v

test-game:
	conda run -n chinese_war_game python -m unittest discover -s tests/game -t . -p "test_*.py" -v

test-integration:
	conda run -n chinese_war_game python -m unittest discover -s tests/integration -t . -p "test_*.py" -v
