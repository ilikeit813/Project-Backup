import sys
import getopt

def main(args):
	print getopt.getopt(args,':')[1][0]
	print getopt.getopt(args,':')[1][1]
if __name__ == "__main__":
        main(sys.argv[1:])
