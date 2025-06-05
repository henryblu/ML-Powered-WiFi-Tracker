idf_component_register(
    SRCS "../time_component_test.c"
    INCLUDE_DIRS ".." "../_components"
    PRIV_REQUIRES unity
)

set_source_files_properties("../time_component_test.c" PROPERTIES LANGUAGE CXX)
