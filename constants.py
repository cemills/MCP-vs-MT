"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

HRS_TO_SECS = 60*60 # seconds per hour
OD_TO_COUNT_CONC = 1e15 # number of cells per OD per meter^3

MODEL_PARAMETER_LIST = ['PermMCPPropanediol','PermMCPPropionaldehyde','PermMCPPropanol','PermMCPPropionyl','PermMCPPropionate',
                        'nmcps',
                        'PermCellPropanediol', 'PermCellPropionaldehyde', 'PermCellPropanol', 'PermCellPropionyl','PermCellPropionate',
                        'VmaxCDEf', 'KmCDEPropanediol',
                        'VmaxPf', 'KmPfPropionaldehyde',
                        'VmaxPr', 'KmPrPropionyl',
                        'VmaxQf', 'KmQfPropionaldehyde',
                        'VmaxQr', 'KmQrPropanol',
                        'VmaxLf', 'KmLPropionyl']


VARIABLE_NAMES = ['PROPANEDIOL_MCP', 'PROPIONALDEHYDE_MCP', 'PROPANOL_MCP', 'PROPIONYL_MCP', 'PROPIONATE_MCP',
                  'PROPANEDIOL_CYTO', 'PROPIONALDEHYDE_CYTO', 'PROPANOL_CYTO', 'PROPIONYL_CYTO', 'PROPIONATE_CYTO',
                  'PROPANEDIOL_EXT', 'PROPIONALDEHYDE_EXT', 'PROPANOL_EXT', 'PROPIONYL_EXT', 'PROPIONATE_EXT']


VARIABLE_INIT_NAMES = ['PROPANEDIOL_MCP_INIT', 'PROPIONALDEHYDE_MCP_INIT', 'PROPANOL_MCP_INIT', 'PROPIONYL_MCP_INIT', 'PROPIONATE_MCP_INIT',
                       'PROPANEDIOL_CYTO_INIT', 'PROPIONALDEHYDE_CYTO_INIT', 'PROPANOL_CYTO_INIT', 'PROPIONYL_CYTO_INIT', 'PROPIONATE_CYTO_INIT',
                       'PROPANEDIOL_EXT_INIT', 'PROPIONALDEHYDE_EXT_INIT', 'PROPANOL_EXT_INIT', 'PROPIONYL_EXT_INIT', 'PROPIONATE_EXT_INIT']


MODEL_PARAMETER_LIST_UNITS = {'PermMCPPropanediol' : "metre/seconds" ,
                              'PermMCPPropionaldehyde' : "metre/seconds" ,
                              'PermMCPPropanol' : "metre/seconds" ,
                              'PermMCPPropionyl' : "metre/seconds" ,
                              'PermMCPPropionate' : "metre/seconds" ,
                              'nmcps': None,
                              'PermCellPropanediol' : "metre/seconds" ,
                              'PermCellPropionaldehyde' : "metre/seconds" ,
                              'PermCellPropanol' : "metre/seconds" ,
                              'PermCellPropionyl' : "metre/seconds" ,
                              'PermCellPropionate' : "metre/seconds" ,
                              'VmaxCDEf' : "millimolar/seconds",
                              'KmCDEPropanediol' : "millimolar",
                              'VmaxPf' : "millimolar/seconds",
                              'KmPfPropionaldehyde' : "millimolar",
                              'VmaxPr' : "millimolar/seconds",
                              'KmPrPropionyl': "millimolar/seconds",
                              'VmaxQf' : "millimolar/seconds",
                              'KmQfPropionaldehyde' : "millimolar",
                              'VmaxQr' : "millimolar/seconds",
                              'KmQrPropanol' : "millimolar",
                              'VmaxLf' : "millimolar/seconds",
                              'KmLPropionyl' : "millimolar",
                              'ncells': None
                              }

VARIABLE_NAMES_UNITS = {'PROPANEDIOL_MCP' : "millimolar",
                        'PROPIONALDEHYDE_MCP' : "millimolar",
                        'PROPANOL_MCP' : "millimolar",
                        'PROPIONYL_MCP' : "millimolar",
                        'PROPIONATE_MCP' : "millimolar",
                        'PROPANEDIOL_CYTO' : "millimolar",
                        'PROPIONALDEHYDE_CYTO' : "millimolar",
                        'PROPANOL_CYTO' : "millimolar",
                        'PROPIONYL_CYTO' : "millimolar",
                        'PROPIONATE_CYTO' : "millimolar",
                        'PROPANEDIOL_EXT' : "millimolar",
                        'PROPIONALDEHYDE_EXT' : "millimolar",
                        'PROPANOL_EXT' : "millimolar",
                        'PROPIONYL_EXT' : "millimolar",
                        'PROPIONATE_EXT' : "millimolar"}

