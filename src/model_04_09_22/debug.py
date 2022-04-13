'''
    Module
    ----------------------------------
    Implements functions used for handling 
    debug mode and config params
'''

import sys

def is_forced():
    ''' Checks for the force flag from stdin

        Returns
        -------
        bool
            state of the force flag
        
        Valid force flags : ['-f', '--force'] (case-sensitive) 

        The force flag will force the script to redo the operations
        even when the data files exist, thereby overwriting them 
    '''
    if len(sys.argv) < 2:
        return False
    if '-f' in sys.argv or '--force' in sys.argv:
        return True
    return False

def is_debug_mode():
    ''' Checks for the debug flag from stdin

        Returns
        -------
        bool
            state of the debug flag
        
        Valid force flags : ['-d', '--debug'] (case-sensitive) 

        The debug flag will allow the script to run the operations
        with debug configuration, thereby constraining the runtime 
    '''
    if len(sys.argv) < 2:
        return False
    if '-d' in sys.argv or '--debug' in sys.argv:
        return True
    return False

def get_debug_config(debug=False):
    ''' Returns the config based on debug mode being turned on or off

    Params
    ------
    debug : bool
        state of debug flag

    Returns
    -------
    config : dict
        a dictionary containing necessary config values
        based on whether debug mode is turned on or off 
    '''
    if debug:
        return {
            'nrows': 100
        }
    return {
        'nrows': None
    }