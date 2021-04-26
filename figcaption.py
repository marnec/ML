import re
from sys import argv
import textwrap

i = 1
figrefs = {}

def insert_figcaption(md, baseurl_subpath=''):
    pattern = r'!\[\w+\]\((.*)\)(\n\s*)+<i id=\"(\S+)\">(.*)</i>'
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

def insert_videocaption(md, baseurl_subpath=''):
    pattern = r'(<video.*>\n(\w|\d|\n|\W)+</video>)(\n\s*)+<i id=\"(\S+)\">(.*)</i>'
    matches = list(re.finditer(pattern, md))

    def repl_and_count(mobj):
        global i
        global figregs

        video, _, _, iid, caption = mobj.groups()
        repl = textwrap.dedent("""
        <figure id="{}">
                {}
            <figcaption>Figure {}. {}</figcaption>
        </figure>""".format(iid, textwrap.indent(video, ' '*12), i, caption))
        figrefs.setdefault(permalink, {})[iid] = i
        i +=1
        return repl

    if matches:
        baseurl = '{{site.baseurl}}' + '/' + baseurl_subpath.lstrip('/').rstrip('/')
        return re.sub(pattern, repl_and_count, md)


def insert_figrefs(md):
    pattern = r'<a href="(ML\d+)?#(fig:\S+)">.*</a>'

    def repl_figrefs(mobj):
        global figrefs
        # iid = mobj.group(1)
        plink, iid = mobj.groups()
        # ref = iid.split("#")
        
        if plink:
            # plink, iid = ref
            href = '{{{{site.basurl}}}}/ML/{}#{}'.format(plink, iid) 
            
        else:
            plink = permalink
            # iid = ref[-1]
            href = '#{}'.format(iid)

        return "<a href=\"{}\">Figure {}</a>".format(href, figrefs[plink][iid])

    return re.sub(pattern, repl_figrefs, md)


def insert_pagerefs(md):
    pattern = r'<a href="page:(ML\d+)(#\S+)?">.*</a>'

    def repl_pagerefs(mobj):
        print(mobj.groups())
        plink, anchor = mobj.groups()
        href = '{{{{site.basurl}}}}/ML/{}'.format(plink)

        if anchor:
            href += anchor
        return '<a href="{}">{}</a>'.format(href, plink)

    return re.sub(pattern, repl_pagerefs, md)


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
        md = update_md(md, insert_pagerefs(md))
        md = update_md(md, insert_figcaption(md, 'pages'))
        md = update_md(md, insert_videocaption(md))
        md = update_md(md, insert_figrefs(md))

        if md:
            with open(mdfile, 'w') as markdown_file:
                markdown_file.write(md)
