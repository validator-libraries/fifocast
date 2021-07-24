from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='fifocast',
  version='0.0.1',
  description='Data Science library python',
  long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='shugaibov-valy',
  author_email='shugaibov2006@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='fifocast', 
  packages=find_packages(),
  install_requires=[''] 
)