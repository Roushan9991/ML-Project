from setuptools import find_packages, setup

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> list[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.strip() for req in requirements]  # ✅ handles \n, \r\n, spaces

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="ML Project",
    version="0.0.1",
    author="Roushan",
    author_email="rkmourya999@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)