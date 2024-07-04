# Logger
code in `logging_de.py`

Logging of every launch and every logic step of the algorithm from `differential_evolution.py`

- Logs with levels 1, 2, 3 are saved in `logging_de.log`

- If result of the algorithm is greater than 1e-3, the status is ERROR, if result is greater than 1e-1, the status is CRITICAL.

- ERROR and CRITICAL are saved in `errors.log`

- Formatter has these features:
  - time of logging 
  - name of logger
  - level of logging
  - action that was done

# Test coverage
code in `test_de.py`

- 100% [test coverage](https://www.atlassian.com/ru/continuous-delivery/software-testing/code-coverage)

- Run test command:

pytest -s test_de.py --cov-report=json --cov
