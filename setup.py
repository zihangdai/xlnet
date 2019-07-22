import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xlnet",
    version="0.0.1",
    author="zihangdai",
    author_email="zander.dai@gmail.com",
    description="XLNet: Generalized Autoregressive Pretraining for Language Understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zihangdai/xlnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
)