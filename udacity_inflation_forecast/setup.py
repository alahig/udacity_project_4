from setuptools import setup, find_packages

setup(
    name="udacity_inflation_forecast",
    version="0.1.0",
    description="Udacity project inflation forecast",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "pandas"],
)
