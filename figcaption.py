import re
from sys import argv
import textwrap
import inspect

i = 1
figrefs = {}

def insert_figcaption(md, baseurl_subpath=''):
    print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
    # pattern = r'!\[\w+\]\((.*)\)(\n\s*)+<i id=\"(\S+)\">(.*)</i>'
    pattern = r'!\[\w+\]\(([^\)]*)\)[\n\s]*<i id=\"([^\"]*)\">(.*)</i>'
    matches = list(re.finditer(pattern, md))


    def repl_and_count(mobj):
        print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
        global i
        global figregs
        url, iid, caption = mobj.groups()
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
    print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
    pattern = r'(<video[^\n]+>[^§]*</video>)(\n\s*)+<i id=\"(\S+)\">(.*)</i>'
    matches = list(re.finditer(pattern, md))

    def repl_and_count(mobj):
        print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
        global i
        global figregs

        video, _, iid, caption = mobj.groups()
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

def insert_scriptcaption(md, baseurl_subpath=''):
    print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
    pattern = r'(<script[^\n]+>[^§]*</script>)(\n\s*)+<i id=\"(\S+)\">(.*)</i>'
    matches = list(re.finditer(pattern, md))

    def repl_and_count(mobj):
        print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
        global i
        global figregs

        video, _, iid, caption = mobj.groups()
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
    print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
    pattern = r'<a href="(ML\d+)?#(fig:\S+)">.*</a>'

    def repl_figrefs(mobj):
        print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
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
    print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
    # pattern = r'<a href="page:(ML\d+)(#\S+)?">.*</a>'
    pattern = r'<a href=\"(page:[^\"]+)\">(.+?)</a>'

    def repl_pagerefs(mobj):
        print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
        plink, anchor = mobj.groups()
        plink = plink.split(':')[1]
        href = '{{{{site.basurl}}}}/ML/{}'.format(plink)

        if anchor:
            href += anchor
        return '<a href="{}">{}</a>'.format(href, plink)

    return re.sub(pattern, repl_pagerefs, md)


def update_md(md, func_call):
    print(f'[{argv[0]}] {inspect.currentframe().f_code.co_name}')
    if func_call is not None:
        md = func_call
    return md


if __name__ == "__main__":
    getnum = lambda fn: fn.split('/')[-1].split('-')[:-1]

    for mdfile in sorted(argv[1:], key=lambda fn: float(getnum(fn)[1])):

        print(f'[{argv[0]}] input file: {mdfile}')
        permalink = ''.join(getnum(mdfile))
        with open(mdfile) as f:
            md = f.read()
        md = update_md(md, insert_pagerefs(md))
        md = update_md(md, insert_figcaption(md, 'pages'))
        md = update_md(md, insert_videocaption(md))
        md = update_md(md, insert_scriptcaption(md))
        md = update_md(md, insert_figrefs(md))

        if md:
            with open(mdfile, 'w') as markdown_file:
                markdown_file.write(md)
        # break