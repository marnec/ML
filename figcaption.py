import re
from sys import argv
import textwrap

i = 1
figrefs = {}

def insert_figcaption(md, baseurl_subpath=''):
    pattern = r'!\[png\]\((.*)\)(\n\s*)+<i id="(\S+)">(.*)</i>'
    matches = list(re.finditer(pattern, md))

    def repl_and_count(mobj):
        global i
        global figregs

        url, _, iid, caption = mobj.groups()
        repl = textwrap.dedent("""
        <figure id="{}">
            <img src="{}/{}" alt="png">
            <figcaption>Figure {}. {}</figcaption>
        </figure>""".format(iid, baseurl, url, i, caption))

        figrefs[iid] = i
        i +=1
        return repl

    if matches:
        baseurl = '{{site.baseurl}}' + '/' + baseurl_subpath.lstrip('/').rstrip('/')
        return re.sub(pattern, repl_and_count, md)


def insert_figrefs(md):

    def repl_figrefs(mobj):
        global figrefs
        iid = mobj.group(1)
        return "<a href=\"{}\">Figure {}</a>".format(iid, figrefs[iid.strip("#")])

    return re.sub(r'<a href="(\S+)">.*</a>', repl_figrefs, md)


def update_md(md, func_call):
    if func_call is not None:
        md = func_call
    return md


if __name__ == "__main__":
    for mdfile in argv[1:]:
        print(mdfile)
        with open(mdfile) as f:
            md = f.read()

        md = update_md(md, insert_figcaption(md, 'pages'))
        md = update_md(md, insert_figrefs(md))

        if md:
            with open(mdfile, 'w') as markdown_file:
                markdown_file.write(md)
