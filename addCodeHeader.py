from sys import argv
import inspect
import re

def addCodeHeader(md: str):
    print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')

    return md.replace('```python', '{% include codeHeader.html %}\n```python')

if __name__ == '__main__':
    getnum = lambda fn: fn.split('/')[-1].split('-')[:-1]

    for mdfile in sorted(argv[1:], key=lambda fn: float(getnum(fn)[1])):

            print(f'[{argv[0]}] input file: {mdfile}')
            with open(mdfile) as f:
                md = f.read()

            md = addCodeHeader(md)

            if md:
                with open(mdfile, 'w') as markdown_file:
                    markdown_file.write(md)