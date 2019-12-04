###################################################################################################
# ---------------------------------------------------------------------------------------
# File:   vision_robot_test.robot
# Author: Luis Monteiro
#
# Created on nov 29, 2019, 22:00 PM
# ---------------------------------------------------------------------------------------
###################################################################################################

# #################################################################################################
# ---------------------------------------------------------------------------------------
*** Settings ***
# ---------------------------------------------------------------------------------------
# #################################################################################################
Documentation  Vision Test Cases Samples
Library   Remote   ${vision_server_ip}   WITH NAME   vision

# #################################################################################################
# ---------------------------------------------------------------------------------------
*** Variables ***
# ---------------------------------------------------------------------------------------
# #################################################################################################
${vision_server_ip}  127.0.0.1:20010

# #################################################################################################
# ---------------------------------------------------------------------------------------
*** Keywords ***
# ---------------------------------------------------------------------------------------
# #################################################################################################
Enable Vision Filters
  [Documentation]   Enable a Group of Filters
  [Arguments]  @{names}
  FOR  ${name}  IN  @{names}
    vision.set_filter  ${name}
  END

Disable Vision Filters
  [Documentation]   Clear All Filters
  vision.clear_filters

Wait Until Vision Number Variable 
  [Documentation]   Read Vision Variable
  [Arguments]  ${name}  ${timeout}  ${value}  ${tolerance}
  [return]     ${result}
  ${value_min}=  Evaluate   ${value} - ${tolerance}
  ${value_max}=  Evaluate   ${value} + ${tolerance}
  ${result}=  vision.get_number_variable  ${name}  ${None}  ${timeout}  ${value_min}  ${value_max}

Wait Until Vision Text Variable 
  [Documentation]   Read Vision Variable
  [Arguments]  ${name}  ${timeout}  ${pattern}
  [return]     ${result}
  ${result}=  vision.get_text_variable  ${name}  ${None}  ${timeout}  ${pattern} 

# #################################################################################################
# ---------------------------------------------------------------------------------------
*** Test Cases ***
# ---------------------------------------------------------------------------------------
# #################################################################################################
# ---------------------------------------------------------------------------------------
# Text Regonition
# ---------------------------------------------------------------------------------------
Text Recognition
  [Documentation]  Text Recognition
  [Tags]  Test
  [Setup]     Enable Vision Filters  speed
  [Teardown]  Disable Vision Filters 
  ${result}=  Wait Until Vision Text Variable  speed  5  .*
  Log To Console  ${result}

###################################################################################################
# ---------------------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------------------
###################################################################################################

