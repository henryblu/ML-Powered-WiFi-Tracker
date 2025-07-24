#include "unity.h"
#include "../../_components/time_component.h"
#include <unistd.h>

TEST_CASE("match_set_timestamp_template valid", "[time]")
{
    TEST_ASSERT_TRUE(match_set_timestamp_template((char *)"SETTIME: 123.456"));
}

TEST_CASE("match_set_timestamp_template invalid", "[time]")
{
    TEST_ASSERT_FALSE(match_set_timestamp_template((char *)"BAD"));
    TEST_ASSERT_FALSE(match_set_timestamp_template((char *)"SETTIME 123.456"));
}

TEST_CASE("time_set sets flag and timestamps increase", "[time]")
{
    real_time_set = false;
    double start = get_system_clock_timestamp();
    usleep(1000);
    time_set((char *)"SETTIME: 1.2");
    TEST_ASSERT_TRUE(real_time_set);
    double after = get_system_clock_timestamp();
    TEST_ASSERT_TRUE(after >= start);
    double t1 = get_system_clock_timestamp();
    usleep(1000);
    double t2 = get_system_clock_timestamp();
    TEST_ASSERT_TRUE(t2 > t1);
}

void app_main(void)
{
    UNITY_BEGIN();
    unity_run_all_tests();
    UNITY_END();
}
