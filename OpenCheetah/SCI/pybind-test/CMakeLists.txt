set(SCI_BACKEND SCI-OT)

pybind11_add_module(sci_provider_alice Binder.cpp)
target_compile_definitions(sci_provider_alice PRIVATE PARTY_ALICE)
target_link_libraries(sci_provider_alice PRIVATE gemini SCI-Cheetah) 
target_compile_options(sci_provider_alice PUBLIC "-Wno-parentheses-equality")
target_compile_options(sci_provider_alice PUBLIC "-fPIC")
target_compile_definitions(sci_provider_alice PUBLIC SCI_OT=1 USE_CHEETAH=1)

pybind11_add_module(sci_provider_bob Binder.cpp)
target_compile_definitions(sci_provider_bob PRIVATE PARTY_BOB)
target_link_libraries(sci_provider_bob PRIVATE gemini SCI-Cheetah) 
target_compile_options(sci_provider_bob PUBLIC "-Wno-parentheses-equality")
target_compile_options(sci_provider_bob PUBLIC "-fPIC")
target_compile_definitions(sci_provider_bob PUBLIC SCI_OT=1 USE_CHEETAH=1)



pybind11_add_module(sci_provider_alice_41 Binder.cpp)
target_compile_definitions(sci_provider_alice_41 PRIVATE PARTY_ALICE BIT41)
target_link_libraries(sci_provider_alice_41 PRIVATE gemini SCI-Cheetah) 
target_compile_options(sci_provider_alice_41 PUBLIC "-Wno-parentheses-equality")
target_compile_options(sci_provider_alice_41 PUBLIC "-fPIC")
target_compile_definitions(sci_provider_alice_41 PUBLIC SCI_OT=1 USE_CHEETAH=1 BIT_LENGTH=41)

pybind11_add_module(sci_provider_bob_41 Binder.cpp)
target_compile_definitions(sci_provider_bob_41 PRIVATE PARTY_BOB BIT41)
target_link_libraries(sci_provider_bob_41 PRIVATE gemini SCI-Cheetah) 
target_compile_options(sci_provider_bob_41 PUBLIC "-Wno-parentheses-equality")
target_compile_options(sci_provider_bob_41 PUBLIC "-fPIC")
target_compile_definitions(sci_provider_bob_41 PUBLIC SCI_OT=1 USE_CHEETAH=1 BIT_LENGTH=41)

# add_executable(test-alice test.cpp)
# target_compile_definitions(test-alice PRIVATE PARTY_ALICE)
# target_link_libraries(test-alice ${SCI_BACKEND}) 
# target_compile_options(test-alice PUBLIC "-Wno-parentheses-equality")

# add_executable(test-bob test.cpp)
# target_compile_definitions(test-bob PRIVATE PARTY_BOB)
# target_link_libraries(test-bob ${SCI_BACKEND}) 
# target_compile_options(test-bob PUBLIC "-Wno-parentheses-equality")
