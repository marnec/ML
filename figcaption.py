import re
from sys import argv
import textwrap

i = 1

def insert_figcaption(infile, baseurl_subpath=''):
	with open(infile) as markdown_file:
		md = markdown_file.read()

	pattern = r'!\[png\]\((.*)\)\n{1,3}<a id="figcaption">(.*)</a>'
	matches = list(re.finditer(pattern, md))
	
	def repl_and_count(mobj):
		global i
		groups = mobj.groups()
		repl = textwrap.dedent("""
			<figure id="figure{}">
				<img src="{}/{}" alt="png">
				<figcaption>Figure {}. {}</figcaption>
			</figure>
			""".format(i, baseurl, groups[0], i, groups[1]))
		
		i +=1
		return repl
	
	if matches:
		baseurl = '{{site.baseurl}}' + '/' + baseurl_subpath.lstrip('/').rstrip('/')
		return re.sub(pattern, repl_and_count, md)

if __name__ == "__main__":
	for mdfile in argv[1:]:
		md = insert_figcaption(mdfile, 'pages')
		
		if md:
			with open(mdfile, 'w') as markdown_file:
				markdown_file.write(md)