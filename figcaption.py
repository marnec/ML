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
        figrefs.setdefault(permalink, {})[iid] = i
        i +=1
        return repl

    if matches:
        baseurl = '{{site.baseurl}}' + '/' + baseurl_subpath.lstrip('/').rstrip('/')
        return re.sub(pattern, repl_and_count, md)


def insert_figrefs(md):

    def repl_figrefs(mobj):
        global figrefs
        iid = mobj.group(1)
        ref = iid.split("#")
        
        if ref[0]:
            plink, iid = ref
            href = '{{{{site.basurl}}}}/ML/{}#{}'.format(plink, iid) 
            
        else:
            plink = permalink
            iid = ref[-1]
            href = '#{}'.format(iid)

        return "<a href=\"{}\">Figure {}</a>".format(href, figrefs[plink][iid])

    return re.sub(r'<a href="(\S+)">.*</a>', repl_figrefs, md)


def update_md(md, func_call):
    if func_call is not None:
        md = func_call
    return md


if __name__ == "__main__":
    getnum = lambda fn: fn.split('/')[-1].split('-')[:-1]
    for mdfile in sorted(argv[1:], key=lambda fn: int(getnum(fn)[1])):

        print(mdfile)
        permalink = ''.join(getnum(mdfile))
        with open(mdfile) as f:
            md = f.read()

        md = update_md(md, insert_figcaption(md, 'pages'))
        md = update_md(md, insert_figrefs(md))

        if md:
            with open(mdfile, 'w') as markdown_file:
                markdown_file.write(md)
