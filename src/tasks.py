"""
Module with invoke tasks
"""

import invoke

import local_invoke.visualize


# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(local_invoke.visualize)
