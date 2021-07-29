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
  version='0.0.5',
  description='Data Science library on python',
  long_description=open('README.md').read(),
  url='',  
  author='shugaibov-valy',
  author_email='shugaibov2006@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='fifocast', 
  package_data={'': ['Data.csv', 'forest_fire.jpg'], 'images_graphs': ['V_type_graph.png', 'V_type_graph_month.png', 'V_type_graph_week.png']},
  packages=find_packages(),
  install_requires=['requests', 'pillow', 'plotly', 'telegraph', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'kaleido'] 
)