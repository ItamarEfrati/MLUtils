import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='MLUtils',
    version='0.0.1',
    author='Itamar Efrati',
    author_email='itamar.efr@gmail.com',
    description='Machine learning utils',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["MLUtils.examples"]),
    url='https://github.com/ItamarEfrati/MLUtils',
    project_urls={
        "Bug Tracker": "https://github.com/ItamarEfrati/MLUtils/issues"
    },
    license='MIT',
)
