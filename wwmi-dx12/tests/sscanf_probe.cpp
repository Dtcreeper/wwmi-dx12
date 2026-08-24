// Temporary diagnostic: verify MSVC sscanf_s('%16llx%n') semantics for
// 0x-prefixed hashes (mirrors 3DMigoto GetIniHash). Remove after use.
#include <cstdio>
#include <cstring>

int main()
{
	unsigned long long h = 0;
	int len = 0;

	const char *cases[] = {
		"0x1234abcd",
		"deadBEEF",
		"0x1234567890abcdef",
		"1234567890abcdef",
		"0x1234567890abcdef0",
		"-1",
		"0x",
		"0x12 zz",
		"1234567890abcdef1",
	};

	for (const char *c : cases)
	{
		h = 0xdead;
		len = -1;
		const int ret = sscanf_s(c, "%16llx%n", &h, &len);
		const size_t sl = strlen(c);
		std::printf("%-22s ret=%d len=%d strlen=%zu h=0x%016llx verdict=%s\n",
			c, ret, len, sl, h,
			(ret == 1 && len == static_cast<int>(sl)) ? "ACCEPT" : "reject");
	}
	return 0;
}
