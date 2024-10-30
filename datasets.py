
THE_PARAMS = {
    
    
    # # for hardening
    # "California": {
    #             "xgb": {
    #                 "n_estimators": 25,
    #                 "max_depth": 4,
    #                 "learning_rate": 0.9,
    #             },
    #             "rf": {
    #                 "n_estimators": 25,
    #                 "max_depth": 10,
    #             },
    #             "delta":  {
    #                         "xgb": 0.03,
    #                         "rf": 0.06, 
    #             }
    # },

    "Covtype": {
                "xgb": {
                    "n_estimators": 50, #2
                    "max_depth": 6, #5,
                    "learning_rate": 0.9,
                },
                "rf": {
                    "n_estimators": 50, 
                    "max_depth": 10, 
                },
                "groot": {
                    "n_estimators": 50, 
                    "max_depth": 10, 
                    "epsilon": 0.01,
                },
                "delta": {
                            "xgb": 0.1,
                            "rf": 0.3,
                            "groot": 0.4, 
                }
        },

    "FashionMnistLt5": {
                "xgb": {
                    "n_estimators": 50,
                    "max_depth": 6, 
                    "learning_rate": 0.1,
                },
                "rf": {
                    "n_estimators": 50, 
                    "max_depth": 10, 
                },
                "groot": {
                    "n_estimators": 50, 
                    "max_depth": 10,
                    "epsilon": 0.3,
                },
                "delta": {
                            "xgb": 0.3, #0.1 
                            "rf": 0.3,
                            "groot": 0.4, 
                    }
    },


    "AtlasHiggs": {
                "xgb": {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                },
                "rf": {
                    "n_estimators": 50, 
                    "max_depth": 10,
                },
                "groot": {
                    "n_estimators": 50, 
                    "max_depth": 10,
                    "epsilon": 0.01, # note: bad performance
                },
                "delta": {
                            "xgb": 0.08, 
                            "rf": 0.08,
                            "groot": 0.4, 
                }
        },


    "MiniBooNE": {
                "xgb": {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                },
                "rf": {
                    "n_estimators": 50,
                    "max_depth": 10, 
                },
                "groot": {
                    "n_estimators": 50,
                    "max_depth": 10,
                    "epsilon": 0.01,
                },
                "delta": {
                            "xgb": 0.08,
                            "rf": 0.08,
                            "groot": 0.5,
                }
        },

    # # for hardening
    # "Mnist2v6": {
    #             "xgb": {
    #                 "n_estimators": 25,
    #                 "max_depth": 3,
    #                 "learning_rate": 0.5,
    #             },
    #             # "rf": {
    #             #     "n_estimators": 50,
    #             #     "max_depth": 10,
    #             # },
    #             "delta": {
    #                         "xgb": 0.3,# 0.4, 
    #                         #"rf": 0.3 
    #                 }
    # },
    
    "MnistLt5": {
                "xgb": {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "learning_rate": 0.5,
                },
                "rf": {
                    "n_estimators": 50,
                    "max_depth": 10,
                },
                "groot": {
                    "n_estimators": 50,
                    "max_depth": 10,
                    "epsilon": 0.3,
                },
                "delta": {
                            "xgb": 0.3,
                            "rf": 0.3,
                            "groot": 0.4,
                    }
    },

    # # for hardening
    # "Phoneme": {
    #             "xgb": {
    #                 "n_estimators": 100, #30,
    #                 "max_depth": 5,
    #                 "learning_rate": 0.5,
    #             },
    #             # "rf": {
    #             #     "n_estimators": 50,
    #             #     "max_depth": 10, 
    #             # },
    #             "delta": {
    #                         "xgb": 0.1,
    #                         "rf": 0.2
    #                 }
    # },


    "Prostate": {
                "xgb": {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "learning_rate": 0.5,
                },
                "rf": {
                    "n_estimators": 50,
                    "max_depth": 10, 
                },
                "groot": {
                    "n_estimators": 50,
                    "max_depth": 10,
                    "epsilon": 0.01,
                },
                "delta": {
                            "xgb": 0.1,
                            "rf": 0.2,
                            "groot": 0.2,
                    }
    },

     "RoadSafety": {
                "xgb": {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "learning_rate": 0.5,
                },
                "rf": {
                    "n_estimators": 50,
                    "max_depth": 10,
                },
                "groot": {
                    "n_estimators": 50,
                    "max_depth": 10,
                    "epsilon": 0.05,
                },
                "delta": {
                                "xgb": 0.06, 
                                "rf": 0.12,
                                "groot": 0.2, 
                    }
    },

    "SensorlessDriveDiagnosisLt6": {
                "xgb": {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "learning_rate": 0.5,
                },
                "rf": {
                    "n_estimators": 50,
                    "max_depth": 10,
                },
                "groot": {
                    "n_estimators": 50,
                    "max_depth": 10,
                    "epsilon": 0.01,
                },
                "delta": {
                                "xgb": 0.06, 
                                "rf": 0.12,
                                "groot": 0.2, 
                    }
    },

    "VehicleSensIt": {
                "xgb": {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                },
                "rf": {
                    "n_estimators": 50,
                    "max_depth": 10,
                },
                "groot": {
                    "n_estimators": 50,
                    "max_depth": 10,
                    "epsilon": 0.1,
                },
                "delta": {
                                "xgb": 0.15, 
                                "rf": 0.15, 
                                "groot": 0.4,
                    }
    },

    "Webspam": {
                "xgb": {
                    "n_estimators": 50,
                    "max_depth": 5,
                    "learning_rate": 0.5,
                },
                "rf": {
                    "n_estimators": 50,
                    "max_depth": 10,
                },
                "groot": {
                    "n_estimators": 50,
                    "max_depth": 10,
                    "epsilon": 0.01,
                },
                "delta": {
                                "xgb": 0.04, 
                                "rf": 0.06,
                                "groot": 0.1,
                    }
    },    
    
}

# TODO: use these as displayed names in plots..?
THE_DATASETS  = {
    
    "Covtype": 'covtype',
    "FashionMnist[Lt5]": 'fmnist',
    "AtlasHiggs": 'higgs',
    "MiniBooNE": 'miniboone',
    # "Mnist2v6": 'mnist2v6',
    "Mnist[Lt5]": 'mnist',
    'Prostate': 'prostate',
    'RoadSafety': 'roadsafety',
    'SensorlessDriveDiagnosis[Lt6]': 'sensorless',
    'VehicleSensIt': 'vehicle',
    'Webspam': 'webspam'
    
}



