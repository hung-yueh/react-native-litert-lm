require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "react-native-litert-lm"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]
  s.platforms    = { :ios => "15.1" }
  s.module_name  = "LiteRTLM"
  s.source       = { :git => package["repository"]["url"], :tag => "#{s.version}" }
  
  s.swift_version = '5.9'

  s.source_files = [
    # Swift, Objective-C/C++ implementation & autolinking
    "ios/*.{swift,m,mm}",
    # Nitrogen generated iOS bridge & shared C++ interfaces
    "nitrogen/generated/ios/**/*.{mm,swift}",
    "nitrogen/generated/shared/c++/**/*.{hpp,cpp}",
  ]

  # Prebuilt LiteRT-LM C engine (xcframework). Not shipped in the npm tarball;
  # fetched on install by scripts/postinstall.js (asset/tag defined in
  # scripts/framework-source.js). Manual/upstream fallback:
  # scripts/download-ios-frameworks.sh.
  s.vendored_frameworks = 'ios/Frameworks/CLiteRTLM.xcframework'

  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++20',
    'SWIFT_VERSION' => '5.9',
    'HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/nitrogen/generated/shared/c++"',
      '"$(PODS_TARGET_SRCROOT)/nitrogen/generated/ios"',
    ].join(' '),
    'OTHER_LDFLAGS' => '$(inherited) -ObjC',
  }

  # Load nitrogen autolinking
  load 'nitrogen/generated/ios/LiteRTLM+autolinking.rb'
  add_nitrogen_files(s)

  # Core React Native dependencies
  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  s.dependency 'ReactCommon/turbomodule/core'

  # Apple frameworks needed by LiteRT-LM engine
  # Metal/MPS: GPU inference, Accelerate: BLAS/LAPACK, CoreML: delegate
  s.frameworks = ['Metal', 'MetalPerformanceShaders', 'Accelerate', 'CoreML', 'CoreGraphics']
  s.libraries = ['c++']

  s.test_spec 'Tests' do |test_spec|
    test_spec.source_files = 'ios/Tests/**/*.{swift}'
  end

  install_modules_dependencies(s)
end

