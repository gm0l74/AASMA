#!/usr/bin/env python3
#---------------------------------
# AASMA
# File : aasma.py
#
# @ start date          22 04 2020
# @ last update         25 04 2020
#---------------------------------

#---------------------------------
# Commands
#---------------------------------
# List of commands:
# - help
# - environment
# - agent

#---------------------------------
# Imports
#---------------------------------
import sys, os, subprocess
import texttable as tt

#---------------------------------
# Constants
#---------------------------------
# Path of aasma's root
ROOT_PATH = os.path.realpath(__file__)
# Remove filename
ROOT_PATH = ROOT_PATH[:len(ROOT_PATH) - 9]

#---------------------------------
# class Command [Abstract]
#---------------------------------
class Command:
    @property
    def name(self):
        raise ValueError('Override \'name\'')

    def usage(self):
        raise ValueError('Override \'usage\'')

    def execute(self, args):
        raise ValueError('Override \'execute\'')

#---------------------------------
# class HelpCommand
#---------------------------------
class HelpCommand(Command):
    @property
    def name(self):
        return 'help'

    def usage(self):
        return 'aasma help'

    def execute(self, args):
        print("Executing 'aasma help'...")
        # Error handle the input
        # (this input is not created by the user)
        if not isinstance(args, dict):
            raise ValueError('Invalid type of arg')

        # Create the table
        tab = tt.Texttable()
        tab.header(['Command', 'Usage'])

        for k, v in args.items():
            tab.add_row([k, v.usage()])

        print(tab.draw())

#---------------------------------
# class EnvironmentCommand
#---------------------------------
class EnvironmentCommand(Command):
    @property
    def name(self):
        return 'environment'

    def usage(self):
        return 'aasma environment'

    def execute(self, args):
        print("Executing 'aasma environment'...")
        # Error handle the input
        if len(args) != 0:
            raise ValueError('Incorrect number of args')

        # Call another python script
        _path = ROOT_PATH + '/aasma/env/run.py'
        subprocess.call("python3 \"{}\"".format(_path), shell=True)

#---------------------------------
# class AgentCommand
#---------------------------------
class AgentCommand(Command):
    @property
    def name(self):
        return 'agent'

    def usage(self):
        return 'aasma agent'

    def execute(self, args):
        print("Executing 'aasma agent'...")
        # Error handle the input
        if len(args) != 0:
            raise ValueError('Incorrect number of args')

        # Call another python script
        _path = ROOT_PATH + '/aasma/agent/run.py'
        subprocess.call("python3 \"{}\"".format(_path), shell=True)

#---------------------------------
# function process_command
#---------------------------------
def process_command():
    # Get arguments
    args = sys.argv

    # Error handle them
    if len(args) < 2:
        print("\033[31mError: {}\033[0m".format("Unspecified command"))
        print("Use 'aasma help' to know more about the module")
        exit(-1)

    # Spawn an instance of each command
    commands= {
        'help': HelpCommand(),
        'environment': EnvironmentCommand(),
        'agent': AgentCommand()
    }

    selected = args[1]
    if selected not in commands:
        print("\033[31mError: {}\033[0m".format("Unknown command"))
        print("Use 'aasma help'")
        exit(-1)

    # Try to execute the function binded to that command
    try:
        if selected == 'help':
            commands['help'].execute(commands)
        else:
            print('ARGS: {}'.format(args[2:]))
            commands[selected].execute(args[2:])
    except Exception as exception:
        print("\033[31mError in '{}': {}\033[0m".format(
            selected, exception
        ))
        print("Usage: '{}'".format(commands[selected].usage()))

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    process_command()
