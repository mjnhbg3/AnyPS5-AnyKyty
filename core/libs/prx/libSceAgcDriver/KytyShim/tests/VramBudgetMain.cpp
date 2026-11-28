#include <string_view>

extern "C" int RunKytyVramBudgetTest(int mode);

int main(int argc, char** argv) {
	const int mode = argc == 2 && std::string_view(argv[1]) == "guest-eviction" ? 1
	               : argc == 2 && std::string_view(argv[1]) == "spill"          ? 2
	               : argc == 2 && std::string_view(argv[1]) == "spill-retry"    ? 3
	                                                                            : 0;
	return RunKytyVramBudgetTest(mode);
}
