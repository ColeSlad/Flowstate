execute_process(COMMAND "${PROGRAM}" --dimension 0 RESULT_VARIABLE status ERROR_VARIABLE error)
if(NOT status EQUAL 1 OR NOT error MATCHES "Positive vectors/dimension required")
  message(FATAL_ERROR "Expected invalid-dimension error, got exit ${status}: ${error}")
endif()
