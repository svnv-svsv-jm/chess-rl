# Developer guide

This document is intended for developers who wish to contribute to the project.

## Tools

If you are a [Visual Studio Code](https://code.visualstudio.com/) user, you may also want to install the following VSCode extensions:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python): pretty mandatory. This should also automatically install Pylance.
- [Black formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter): prettu much needed, install it then on VSCode, in Settings, check the box "Editor: Format On Save".
- [Mypy](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker): not only this will force you to code in a readable way, but will often spot bugs early while you're still coding.
- [Pylint](https://marketplace.visualstudio.com/items?itemName=ms-python.pylint): it will spot, while coding, violations of Python coding best practices. It will help you improve your code quality.
- [TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml): in order to have well-colored `.toml` files while editing them (not really needed).

Also, in VSCoce settings, activate the "Editor: Word Wrap" option, and other similar ones. This will allow you to visualize correctly even long lines of code.

## Technologies

- This projct uses `make` commands to facilitate users and developers.

- This project contains a working Docker image. You can build and develop in a container running this image if you experience problems in simply installing this project in a virtual environment. You can launch a development container by running:

```bash
make up SERVICES=dev-container
```

- This project uses [pytest](https://docs.pytest.org/en/7.1.x/) with the following plugins: [pylint](https://pylint.pycqa.org/en/latest/) and [mypy](http://www.mypy-lang.org/). [Coverage](https://coverage.readthedocs.io/en/6.4.4/) is also enforced.

## Project's layout

We have chosen the "src" layout from the official [PyTest doc](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html).

```bash
src/
 |--package_1
 |--package_2
 |-- ...
```

Develop any new package within the `src` folder. For example, if creating a package named `cool`, create the folder `src/cool`. Make sure it contains the `__init__.py` file: `src/cool/__init__.py`. Packages can implement an attack, a dataset, a model, or even everything at the same time.

```bash
src/
 |--cool
    |--__init__.py
 |--package_2
 |-- ...
```

The folder `test` will be used to test the code residing in `src`. The `test` folder will contain the `conftest.py` file and then should mimic the layout of the `src` folder as much as possible for all tests. This way, it will be easy to find a specific test for a certain function residing in specific (sub)module.

Notebooks for quick development or to showcase our code can be stored in `notebooks/`. They may import all modules in `src` if needed.

We can use the `results` folder in case we want our repository to store useful results, figures, etc. although it is not recommended to store huge amount of data here on Gitlab.

Code that can be re-used to run specific experiments can be placed in `experiments/`.

## Dependencies

In order to install this project's dependencies, this project offers the following command:

```bash
make install
```

which uses [Poetry](https://python-poetry.org/) behind the scenes.

### Add new dependencies

To add new dependencies, run:

```bash
poetry add <package-name>
```

or manually edit the `pyproject.toml` file, then run: `poetry lock`.

## Testing

This project uses `pytest` for test-driven development (TDD). Make sure you're familiar with it: <https://docs.pytest.org/en/7.1.x/>

To run all your tests, run:

```bash
python -m pytest
# test notebooks too
pytest --nbmake --overwrite "./notebooks
```

`pylint`, `mypy` and `coverage` may be enforced automatically.

You can also just run:

```bash
make tests
```

## CI/CD Pipeline

We use a `.gitlab-ci.yml` file to automatically test our code when pushing.

We only test the source code here, not the notebooks, or the tests may be too long and we must stay within one hour.
