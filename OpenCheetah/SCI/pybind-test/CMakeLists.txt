set(SCI_BACKEND SCI-OT)

macro (setup_pybind_module _name)
    target_link_libraries(${_name} PRIVATE gemini SCI-Cheetah) 
    target_compile_options(${_name} PUBLIC "-Wno-parentheses-equality")
    target_compile_options(${_name} PUBLIC "-fPIC")
    target_compile_definitions(${_name} PUBLIC SCI_OT=1 USE_CHEETAH=1)
endmacro()

pybind11_add_module(cheetah_provider_alice Binder.cpp)
target_compile_definitions(cheetah_provider_alice PRIVATE PARTY_ALICE)
setup_pybind_module(cheetah_provider_alice)

pybind11_add_module(cheetah_provider_bob Binder.cpp)
target_compile_definitions(cheetah_provider_bob PRIVATE PARTY_BOB)
setup_pybind_module(cheetah_provider_bob)


pybind11_add_module(cheetah_provider_alice_41 Binder.cpp)
target_compile_definitions(cheetah_provider_alice_41 PRIVATE PARTY_ALICE BIT41)
target_compile_definitions(cheetah_provider_alice_41 PUBLIC BIT_LENGTH=41)
setup_pybind_module(cheetah_provider_alice_41)

pybind11_add_module(cheetah_provider_bob_41 Binder.cpp)
target_compile_definitions(cheetah_provider_bob_41 PRIVATE PARTY_BOB BIT41)
target_compile_definitions(cheetah_provider_bob_41 PUBLIC BIT_LENGTH=41)
setup_pybind_module(cheetah_provider_bob_41)



pybind11_add_module(cheetah_provider_alice_37 Binder.cpp)
target_compile_definitions(cheetah_provider_alice_37 PRIVATE PARTY_ALICE BIT37)
target_compile_definitions(cheetah_provider_alice_37 PUBLIC BIT_LENGTH=37)
setup_pybind_module(cheetah_provider_alice_37)

pybind11_add_module(cheetah_provider_bob_37 Binder.cpp)
target_compile_definitions(cheetah_provider_bob_37 PRIVATE PARTY_BOB BIT37)
target_compile_definitions(cheetah_provider_bob_37 PUBLIC BIT_LENGTH=37)
setup_pybind_module(cheetah_provider_bob_37)

# add_executable(test-alice test.cpp)
# target_compile_definitions(test-alice PRIVATE PARTY_ALICE)
# target_link_libraries(test-alice ${SCI_BACKEND}) 
# target_compile_options(test-alice PUBLIC "-Wno-parentheses-equality")

# add_executable(test-bob test.cpp)
# target_compile_definitions(test-bob PRIVATE PARTY_BOB)
# target_link_libraries(test-bob ${SCI_BACKEND}) 
# target_compile_options(test-bob PUBLIC "-Wno-parentheses-equality")
