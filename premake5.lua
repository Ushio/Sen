include "libs/PrLib"

workspace "HogeProject"
    location "build"
    configurations { "Debug", "Release" }
    startproject "main"

architecture "x86_64"

externalproject "prlib"
	location "libs/PrLib/build" 
    kind "StaticLib"
    language "C++"

project "main"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }
    cppdialect "C++14"

    -- Src
    files { "main.cpp", "sen.h" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("Main_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("Main")
        optimize "Full"
    filter{}

    
project "unittest"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }
    cppdialect "C++14"
    buildoptions "/bigobj"

    -- Src
    files { "unittest.cpp", "catch_amalgamated.cpp", "catch_amalgamated.hpp" }
    includedirs { "." }
    includedirs { "libs/eigen-3.4.0"}

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("Unittest_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("Unittest")
        optimize "Full"
    filter{}