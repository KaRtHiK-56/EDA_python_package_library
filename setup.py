from setuptools import setup, find_packages


with open(r"C:\Users\Devadarsan\Desktop\Karthik_projects\EDA_Package\EDA_python_package_library\README.md", "r") as f:
    long_description = f.read()

 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3.10'
]
 
setup(
  name='edapython',
  version='0.0.2.3',
  description='A Library for Making the Explorartory Data Analysis process easy in single line of codes',
  long_description=long_description,
  url="https://github.com/KaRtHiK-56/EDA_python_package_library",  
  author='Karthik',
  author_email='karthiksurya611@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='eda', 
  packages=find_packages(),
  install_requires=[''] 
)