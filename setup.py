from setuptools import find_packages, setup

# Function to read requirements.txt and ignore invalid lines like '-e .'
def get_requirements(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        # Keep only valid package lines
        requirements = [
            req.strip() 
            for req in requirements 
            if req.strip() and not req.strip().startswith("-e")
        ]
    return requirements

# Optional: include README.md as long_description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="browsing_history_recommender",
    version="0.1.0",
    author="Samriddhi Sonker",
    author_email="sonkersamriddhi@gmail.com",
    description="A product recommendation system using browsing history",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.12",
)
