import getopt, sys, re

class ParseArgs:
    '''
    class for parsing input arguments
    '''

    def __init__(self, arg_list: list, help_msg: str):
        '''
        Parameters
        ----------
        arg_list: list
            list of input args, e.g., ['flag', 'in=', 'out=']
        help_msg: str
            help message
        '''
        self.arg_list = arg_list
        self.help_msg = help_msg

        if not 'help' in self.arg_list:
            self.arg_list.append('help')


    def printhelp(self):
        print(self.help_msg)

    def get_args(self, args: list) -> dict:
        '''
        parse arguments

        Parameters
        ----------
        args: [str]
            input arguments, e.g., sys.argv[1:]

        Returns
        -------
        dict of parsed args, e.g., dict {'flag': bool, 'in': str, 'out': str}
            For arguments returning a value, a str will be returned with a
            default value of ''. For flags, a bool will be returned with a
            defualt value of False.
        '''
        try:
            (opts,_) = getopt.getopt(args,'h', self.arg_list)
        except getopt.GetoptError:
            self.printhelp()
            sys.exit()

        parsed = dict()

        # init
        for x in self.arg_list:
            if '=' in x:
                x = re.sub('=', '', x)
                parsed[x] = ''
            else:
                parsed[x] = False

        # parse
        for (opt,arg) in opts:
            if opt in ('-h', '--help'):
                self.printhelp()
                sys.exit()
            for x in self.arg_list:
                if '=' in x:
                    x = re.sub('=', '', x)
                    if opt == '--'+x:
                        parsed[x] = arg
                else:
                    if opt == '--'+x:
                        parsed[x] = True
        return parsed
