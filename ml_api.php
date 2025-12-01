<?php
/**
 * ML + AI API
 * Bridge between frontend and Flask ML + AI service
 */

// Error handling
ini_set('display_errors', 0);
ini_set('log_errors', 1);
error_reporting(E_ALL);

// Set JSON header
header('Content-Type: application/json');

// Get action
$action = $_GET['action'] ?? $_POST['action'] ?? '';

// For API calls, skip session handling completely
if (in_array($action, ['status', 'analyze', 'get_analysis', 'analyze_batch', 'train', 'test', 'start_service'])) {
    // API calls - no session required
    require_once __DIR__ . '/../includes/conn.php';
    require_once __DIR__ . '/../includes/rate_limit_helper.php';
    
    // Rate limiting for API endpoints: 20 requests per minute per IP
    $ipAddress = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
    if (!checkRateLimit('ml_api_' . $action, $ipAddress, 20, 60)) {
        http_response_code(429);
        echo json_encode([
            'status' => 'error',
            'message' => 'Too many requests. Please slow down and try again.'
        ]);
        exit();
    }
} else {
    // For non-API calls, require full session
    require_once __DIR__ . '/../includes/session.php';
}

switch($action) {
    case 'analyze':
        analyze_couple();
        break;
    case 'get_analysis':
        get_existing_analysis();
        break;
    case 'analyze_batch':
        analyze_batch();
        break;
    case 'train':
        train_models();
        break;
    case 'status':
        check_status();
        break;
    case 'start_service':
        start_flask_service();
        break;
    case 'test':
        echo json_encode(['status' => 'success', 'message' => 'API is working']);
        break;
    default:
        echo json_encode(['status' => 'error', 'message' => 'Invalid action']);
}

function analyze_couple() {
    try {
        $access_id = $_GET['access_id'] ?? $_POST['access_id'] ?? null;
        
        if (!$access_id) {
            echo json_encode(['status' => 'error', 'message' => 'access_id required']);
            return;
        }
        
        // Get couple data from database
        $couple_data = get_couple_data($access_id);
        if (!$couple_data) {
            echo json_encode(['status' => 'error', 'message' => 'Couple not found']);
            return;
        }
        
        // Prepare data for ML + AI analysis
        $analysis_data = [
            'access_id' => $access_id,
            'male_age' => $couple_data['male_age'],
            'female_age' => $couple_data['female_age'],
            'civil_status' => $couple_data['civil_status'] ?? 'Single',
            'years_living_together' => $couple_data['years_living_together'] ?? 0,
            'past_children' => $couple_data['past_children'] ?? false,
            'children' => $couple_data['children'] ?? 0,
            'education_level' => $couple_data['education_level'],
            'income_level' => $couple_data['income_level'],
            'questionnaire_responses' => $couple_data['questionnaire_responses'],
            // ADD PERSONALIZED FEATURES FOR NLG
            'male_responses' => $couple_data['male_responses'] ?? [],
            'female_responses' => $couple_data['female_responses'] ?? [],
            'personalized_features' => $couple_data['personalized_features'] ?? [],
            // ADD COUPLE NAMES FOR NLG PERSONALIZATION
            'male_name' => $couple_data['male_name'] ?? 'Male Partner',
            'female_name' => $couple_data['female_name'] ?? 'Female Partner'
        ];
        
        // DEBUG: Log the data being sent to Flask
        error_log("DEBUG - Sending data to Flask for access_id: $access_id");
        error_log("DEBUG - Analysis data: " . json_encode($analysis_data, JSON_PRETTY_PRINT));
        
        // Call Flask service
        $flask_url = 'http://127.0.0.1:5000/analyze';
        $response = call_flask_service($flask_url, $analysis_data, 'POST');
        
        if ($response['status'] === 'success') {
            // Save analysis results to database
            $save_result = save_analysis_results($access_id, $response);
            if ($save_result) {
                echo json_encode($response);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'Failed to save analysis results']);
            }
        } else {
            echo json_encode($response);
        }
        
    } catch (Exception $e) {
        error_log("Analysis error: " . $e->getMessage());
        echo json_encode(['status' => 'error', 'message' => 'Analysis failed: ' . $e->getMessage()]);
    }
}

function get_couple_data($access_id) {
    $conn = get_db_connection();
    
    try {
        // Get couple profile - fix the JOIN condition (no cp.id column exists)
        $stmt = $conn->prepare("
            SELECT 
                cp.first_name, cp.last_name, cp.age, cp.sex,
                cp.civil_status, cp.education, cp.monthly_income,
                cp.years_living_together, cp.past_children, cp.past_children_count,
                cp2.first_name as partner_first_name, cp2.last_name as partner_last_name, 
                cp2.age as partner_age, cp2.sex as partner_sex
            FROM couple_profile cp
            LEFT JOIN couple_profile cp2 ON cp.access_id = cp2.access_id AND cp.sex != cp2.sex
            WHERE cp.access_id = ?
        ");
        
        $stmt->bind_param("s", $access_id);
        $stmt->execute();
        $result = $stmt->get_result();
        
        if ($result->num_rows === 0) {
            return null;
        }
        
        $profiles = $result->fetch_all(MYSQLI_ASSOC);
        
        // Separate male and female profiles
        $male_profile = null;
        $female_profile = null;
        
        foreach ($profiles as $profile) {
            if ($profile['sex'] === 'Male') {
                $male_profile = $profile;
            } else {
                $female_profile = $profile;
            }
        }
        
        if (!$male_profile || !$female_profile) {
            return null;
        }
        
        // Get questionnaire responses from couple_responses table - SEPARATE BY PARTNER
        $stmt = $conn->prepare("
            SELECT cr.response, cr.category_id, cr.question_id, cr.sub_question_id, cr.respondent
            FROM couple_responses cr
            WHERE cr.access_id = ?
            ORDER BY cr.category_id, cr.question_id, cr.sub_question_id, cr.respondent
        ");
        
        $stmt->bind_param("s", $access_id);
        $stmt->execute();
        $result = $stmt->get_result();
        
        // Separate male and female responses
        $male_responses = [];
        $female_responses = [];
        $all_responses = [];
        
        while ($row = $result->fetch_assoc()) {
            // Convert response to numeric value (1-5 scale)
            $response_value = 3; // default neutral
            if (is_numeric($row['response'])) {
                $response_value = (int)$row['response'];
            } else {
                // Handle text responses by mapping to numeric values
                $response_lower = strtolower($row['response']);
                if (strpos($response_lower, 'strongly disagree') !== false || strpos($response_lower, 'never') !== false) {
                    $response_value = 1;
                } elseif (strpos($response_lower, 'disagree') !== false || strpos($response_lower, 'rarely') !== false) {
                    $response_value = 2;
                } elseif (strpos($response_lower, 'neutral') !== false || strpos($response_lower, 'sometimes') !== false) {
                    $response_value = 3;
                } elseif (strpos($response_lower, 'agree') !== false || strpos($response_lower, 'often') !== false) {
                    $response_value = 4;
                } elseif (strpos($response_lower, 'strongly agree') !== false || strpos($response_lower, 'always') !== false) {
                    $response_value = 5;
                }
            }
            
            // Store by partner
            if (strtolower($row['respondent']) === 'male') {
                $male_responses[] = $response_value;
            } else {
                $female_responses[] = $response_value;
            }
            $all_responses[] = $response_value;
        }
        
        // PERSONALIZED ANALYSIS: Calculate partner dynamics
        $personalized_features = calculate_personalized_features($male_responses, $female_responses, $all_responses);
        
        // DEBUG: Log personalized features
        error_log("DEBUG - Access ID: $access_id");
        error_log("DEBUG - Male responses count: " . count($male_responses));
        error_log("DEBUG - Female responses count: " . count($female_responses));
        error_log("DEBUG - Alignment score: " . $personalized_features['alignment_score']);
        error_log("DEBUG - Conflict ratio: " . $personalized_features['conflict_ratio']);
        error_log("DEBUG - Power balance: " . $personalized_features['power_balance']);
        
        // Ensure we have the right number of responses
        if (count($all_responses) > 59) {
            $all_responses = array_slice($all_responses, 0, 59);
        } elseif (count($all_responses) < 59) {
            $all_responses = array_pad($all_responses, 59, 3);
        }
        
        if (empty($all_responses)) {
            $all_responses = array_fill(0, 59, 3);
        }
        
        // Map education and income to numeric levels
        $education_mapping = [
            'No Education' => 0,
            'Pre School' => 0,
            'Elementary Level' => 0,
            'Elementary Graduate' => 0,
            'High School Level' => 1,
            'High School Graduate' => 1,
            'Junior HS Level' => 1,
            'Junior HS Graduate' => 1,
            'Senior HS Level' => 1,
            'Senior HS Graduate' => 1,
            'College Level' => 2,
            'College Graduate' => 3,
            'Vocational/Technical' => 2,
            'ALS' => 1,
            'Post Graduate' => 4
        ];
        
        $income_mapping = [
            '5000 below' => 0,
            '5999-9999' => 0,
            '10000-14999' => 1,
            '15000-19999' => 1,
            '20000-24999' => 2,
            '25000 above' => 3
        ];
        
        // Get civil status (use first profile that has it)
        $civil_status = $male_profile['civil_status'] ?? $female_profile['civil_status'] ?? 'Single';
        
        // Get years living together (only if civil status is "Living In")
        $years_living_together = 0;
        if ($civil_status === 'Living In') {
            $years_living_together = (int)($male_profile['years_living_together'] ?? $female_profile['years_living_together'] ?? 0);
        }
        
        // Get past children info
        $past_children = ($male_profile['past_children'] === 'Yes' || $female_profile['past_children'] === 'Yes');
        $children = 0;
        if ($past_children) {
            $children = (int)($male_profile['past_children_count'] ?? $female_profile['past_children_count'] ?? 0);
        }
        
        return [
            'male_age' => (int)($male_profile['age'] ?? 30),
            'female_age' => (int)($female_profile['age'] ?? 28),
            'civil_status' => $civil_status,
            'years_living_together' => $years_living_together,
            'past_children' => $past_children,
            'children' => $children,
            'education_level' => $education_mapping[$male_profile['education'] ?? 'College Level'] ?? 2,
            'income_level' => $income_mapping[$male_profile['monthly_income'] ?? '10000-14999'] ?? 1,
            'questionnaire_responses' => $all_responses,
            // PERSONALIZED FEATURES
            'male_responses' => $male_responses,
            'female_responses' => $female_responses,
            'personalized_features' => $personalized_features,
            // ADD COUPLE NAMES FOR NLG PERSONALIZATION
            'male_name' => ($male_profile['first_name'] ?? '') . ' ' . ($male_profile['last_name'] ?? ''),
            'female_name' => ($female_profile['first_name'] ?? '') . ' ' . ($female_profile['last_name'] ?? '')
        ];
        
    } catch (Exception $e) {
        error_log("Error getting couple data: " . $e->getMessage());
        return null;
    }
}

function get_personalized_features_for_couple($access_id) {
    // Get couple responses and calculate personalized features
    $conn = new mysqli('localhost', 'root', '', 'u520834156_DBpmoc25');
    if ($conn->connect_error) {
        return [
            'alignment_score' => 0.5,
            'conflict_ratio' => 0.0,
            'power_balance' => 1.0,
            'male_avg_response' => 3.0,
            'female_avg_response' => 3.0,
            'male_consistency' => 0.5,
            'female_consistency' => 0.5,
            'response_variance' => 0.0
        ];
    }
    
    // Get responses for this couple
    $stmt = $conn->prepare("
        SELECT respondent, response
        FROM couple_responses 
        WHERE access_id = ? 
        ORDER BY respondent, question_id
    ");
    $stmt->bind_param("i", $access_id);
    $stmt->execute();
    $result = $stmt->get_result();
    
    $male_responses = [];
    $female_responses = [];
    $all_responses = [];
    
    while ($row = $result->fetch_assoc()) {
        // Convert response to numeric value (1-5 scale)
        $response_value = 3; // default neutral
        if (is_numeric($row['response'])) {
            $response_value = (int)$row['response'];
        } else {
            // Handle text responses by mapping to numeric values
            $response_lower = strtolower($row['response']);
            if (strpos($response_lower, 'strongly disagree') !== false || strpos($response_lower, 'never') !== false) {
                $response_value = 1;
            } elseif (strpos($response_lower, 'disagree') !== false || strpos($response_lower, 'rarely') !== false) {
                $response_value = 2;
            } elseif (strpos($response_lower, 'neutral') !== false || strpos($response_lower, 'sometimes') !== false) {
                $response_value = 3;
            } elseif (strpos($response_lower, 'agree') !== false || strpos($response_lower, 'often') !== false) {
                $response_value = 4;
            } elseif (strpos($response_lower, 'strongly agree') !== false || strpos($response_lower, 'always') !== false) {
                $response_value = 5;
            }
        }
        
        // Store by partner
        if (strtolower($row['respondent']) === 'male') {
            $male_responses[] = $response_value;
        } else {
            $female_responses[] = $response_value;
        }
        $all_responses[] = $response_value;
    }
    
    $conn->close();
    
    // Calculate personalized features
    return calculate_personalized_features($male_responses, $female_responses, $all_responses);
}

function calculate_personalized_features($male_responses, $female_responses, $all_responses) {
    // Calculate relationship dynamics and personalized features
    
    // 1. ALIGNMENT ANALYSIS
    $alignment_score = 0;
    $conflict_count = 0;
    $total_questions = min(count($male_responses), count($female_responses));
    
    for ($i = 0; $i < $total_questions; $i++) {
        $male_resp = $male_responses[$i] ?? 3;
        $female_resp = $female_responses[$i] ?? 3;
        
        // Calculate alignment (how close their responses are)
        $difference = abs($male_resp - $female_resp);
        $alignment_score += (4 - $difference) / 4; // 0-1 scale
        
        // Count conflicts (responses that are very different)
        if ($difference >= 2) {
            $conflict_count++;
        }
    }
    
    $alignment_score = $total_questions > 0 ? $alignment_score / $total_questions : 0.5;
    $conflict_ratio = $total_questions > 0 ? $conflict_count / $total_questions : 0;
    
    // 2. PARTNER-SPECIFIC ANALYSIS
    $male_avg = count($male_responses) > 0 ? array_sum($male_responses) / count($male_responses) : 3;
    $female_avg = count($female_responses) > 0 ? array_sum($female_responses) / count($female_responses) : 3;
    
    // 3. RESPONSE PATTERN ANALYSIS
    $male_consistency = calculate_consistency($male_responses);
    $female_consistency = calculate_consistency($female_responses);
    
    // 4. RELATIONSHIP DYNAMICS
    // Calculate power balance as a ratio (male_avg / female_avg)
    // 1.0 = balanced, >1.0 = male dominance, <1.0 = female dominance
    $power_balance = $female_avg > 0 ? $male_avg / $female_avg : 1.0;
    $response_variance = calculate_variance($all_responses);
    
    return [
        'alignment_score' => $alignment_score,
        'conflict_ratio' => $conflict_ratio,
        'male_avg_response' => $male_avg,
        'female_avg_response' => $female_avg,
        'male_consistency' => $male_consistency,
        'female_consistency' => $female_consistency,
        'power_balance' => $power_balance,
        'response_variance' => $response_variance,
        'total_conflicts' => $conflict_count
    ];
}

function calculate_consistency($responses) {
    if (count($responses) < 2) return 1.0;
    
    $variance = 0;
    $mean = array_sum($responses) / count($responses);
    
    foreach ($responses as $response) {
        $variance += pow($response - $mean, 2);
    }
    
    $variance = $variance / count($responses);
    return max(0, 1 - ($variance / 4)); // 0-1 scale, higher = more consistent
}

function calculate_variance($responses) {
    if (count($responses) < 2) return 0;
    
    $mean = array_sum($responses) / count($responses);
    $variance = 0;
    
    foreach ($responses as $response) {
        $variance += pow($response - $mean, 2);
    }
    
    return $variance / count($responses);
}

function save_analysis_results($access_id, $results) {
    $conn = get_db_connection();
    
    try {
        // Prepare data for saving
        $risk_level = $results['risk_level'] ?? 'Medium';
        $ml_confidence = $results['ml_confidence'] ?? 0;
        $category_scores = json_encode($results['category_scores'] ?? []);
        $focus_categories = json_encode($results['focus_categories'] ?? []);
        $recommendations = json_encode($results['recommendations'] ?? []);
        $analysis_method = $results['analysis_method'] ?? 'Random Forest Counseling Topics Model';
        
        // Insert or update analysis results in ml_analysis table
        $stmt = $conn->prepare("
            INSERT INTO ml_analysis 
            (access_id, risk_level, ml_confidence, category_scores, focus_categories, recommendations, analysis_method, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NOW())
            ON DUPLICATE KEY UPDATE
            risk_level = VALUES(risk_level),
            ml_confidence = VALUES(ml_confidence),
            category_scores = VALUES(category_scores),
            focus_categories = VALUES(focus_categories),
            recommendations = VALUES(recommendations),
            analysis_method = VALUES(analysis_method),
            updated_at = NOW()
        ");
        
        $stmt->bind_param(
            "ssdssss",
            $access_id,
            $risk_level,
            $ml_confidence,
            $category_scores,
            $focus_categories,
            $recommendations,
            $analysis_method
        );
        
        $result = $stmt->execute();
        if ($result) {
            error_log("ML analysis saved to ml_analysis table for access_id: $access_id");
            return true;
        } else {
            error_log("Failed to save ML analysis for access_id: $access_id");
            return false;
        }
        
    } catch (Exception $e) {
        error_log("Error saving ML analysis to ml_analysis table: " . $e->getMessage());
        return false;
    }
}

function get_existing_analysis() {
    try {
        $access_id = $_GET['access_id'] ?? $_POST['access_id'] ?? null;
        
        if (!$access_id) {
            echo json_encode(['status' => 'error', 'message' => 'access_id required']);
            return;
        }
        
        $conn = get_db_connection();
        
        // Fetch existing analysis from database
        $stmt = $conn->prepare("
            SELECT 
                access_id,
                risk_level,
                ml_confidence,
                category_scores,
                focus_categories,
                recommendations,
                analysis_method,
                generated_at,
                updated_at
            FROM ml_analysis
            WHERE access_id = ?
        ");
        
        $stmt->bind_param("s", $access_id);
        $stmt->execute();
        $result = $stmt->get_result();
        
        if ($result->num_rows === 0) {
            echo json_encode([
                'status' => 'success',
                'analyzed' => false,
                'message' => 'No analysis found for this couple'
            ]);
            return;
        }
        
        $row = $result->fetch_assoc();
        
        // Get personalized features for this couple
        $personalized_features = get_personalized_features_for_couple($access_id);
        
        echo json_encode([
            'status' => 'success',
            'analyzed' => true,
            'risk_level' => $row['risk_level'],
            'ml_confidence' => (float)$row['ml_confidence'],
            'category_scores' => json_decode($row['category_scores'], true),
            'focus_categories' => json_decode($row['focus_categories'], true),
            'recommendations' => json_decode($row['recommendations'], true),
            'analysis_method' => $row['analysis_method'],
            'generated_at' => $row['generated_at'],
            'updated_at' => $row['updated_at'],
            // Add personalized features
            'alignment_score' => $personalized_features['alignment_score'],
            'conflict_ratio' => $personalized_features['conflict_ratio'],
            'power_balance' => $personalized_features['power_balance'],
            'male_avg_response' => $personalized_features['male_avg_response'],
            'female_avg_response' => $personalized_features['female_avg_response'],
            'male_consistency' => $personalized_features['male_consistency'],
            'female_consistency' => $personalized_features['female_consistency'],
            'response_variance' => $personalized_features['response_variance']
        ]);
        
        $conn->close();
        
    } catch (Exception $e) {
        error_log("Get analysis error: " . $e->getMessage());
        echo json_encode(['status' => 'error', 'message' => 'Failed to fetch analysis: ' . $e->getMessage()]);
    }
}

function analyze_batch() {
    try {
        // Get list of access_ids to analyze
        $access_ids = $_POST['access_ids'] ?? null;
        
        if (!$access_ids) {
            echo json_encode(['status' => 'error', 'message' => 'access_ids required']);
            return;
        }
        
        // If it's a JSON string, decode it
        if (is_string($access_ids)) {
            $access_ids = json_decode($access_ids, true);
        }
        
        $results = [
            'total' => count($access_ids),
            'success' => 0,
            'failed' => 0,
            'errors' => []
        ];
        
        foreach ($access_ids as $access_id) {
            try {
                // Get couple data
                $couple_data = get_couple_data($access_id);
                
                if (!$couple_data) {
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id not found";
                    continue;
                }
                
                // Prepare data for ML analysis
                $analysis_data = [
                    'access_id' => $access_id,
                    'male_age' => $couple_data['male_age'],
                    'female_age' => $couple_data['female_age'],
                    'civil_status' => $couple_data['civil_status'] ?? 'Single',
                    'years_living_together' => $couple_data['years_living_together'] ?? 0,
                    'past_children' => $couple_data['past_children'] ?? false,
                    'children' => $couple_data['children'] ?? 0,
                    'education_level' => $couple_data['education_level'],
                    'income_level' => $couple_data['income_level'],
                    'questionnaire_responses' => $couple_data['questionnaire_responses']
                ];
                
                // Call Flask service
                $flask_url = 'http://127.0.0.1:5000/analyze';
                $response = call_flask_service($flask_url, $analysis_data, 'POST');
                
                if ($response && isset($response['status']) && $response['status'] === 'success') {
                    // Save analysis results
                    $save_result = save_analysis_results($access_id, $response);
                    if ($save_result) {
                        $results['success']++;
                    } else {
                        $results['failed']++;
                        $results['errors'][] = "Couple $access_id: Failed to save results";
                    }
                } else {
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: " . ($response['message'] ?? 'Unknown error');
                }
                
            } catch (Exception $e) {
                $results['failed']++;
                $results['errors'][] = "Couple $access_id: " . $e->getMessage();
                error_log("Batch analysis error for $access_id: " . $e->getMessage());
            }
        }
        
        echo json_encode([
            'status' => 'success',
            'message' => "Analyzed {$results['success']} of {$results['total']} couples",
            'results' => $results
        ]);
        
    } catch (Exception $e) {
        error_log("Batch analysis error: " . $e->getMessage());
        echo json_encode(['status' => 'error', 'message' => 'Batch analysis failed: ' . $e->getMessage()]);
    }
}

function train_models() {
    try {
        $flask_url = 'http://127.0.0.1:5000/train';
        $response = call_flask_service($flask_url, [], 'POST');
        echo json_encode($response);
    } catch (Exception $e) {
        echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
    }
}

function check_status() {
    try {
        $flask_url = 'http://127.0.0.1:5000/status';
        $response = call_flask_service($flask_url, [], 'GET');
        echo json_encode($response);
    } catch (Exception $e) {
        echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
    }
}

function start_flask_service() {
    try {
        // Path to the PowerShell script
        $script_path = __DIR__ . '\\start_service.ps1';
        
        // Check if the script exists
        if (!file_exists($script_path)) {
            echo json_encode([
                'status' => 'error', 
                'message' => 'start_service.ps1 not found. Please ensure the script exists in ml_model folder.'
            ]);
            return;
        }
        
        // Check if Flask is already running
        $status_url = 'http://127.0.0.1:5000/status';
        $status_response = call_flask_service($status_url, [], 'GET');
        
        if ($status_response['status'] === 'success') {
            echo json_encode([
                'status' => 'success',
                'message' => 'Flask service is already running',
                'already_running' => true
            ]);
            return;
        }
        
        // Start the Flask service using PowerShell in the background
        // Use -WindowStyle Hidden to run in background
        $command = "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File \"$script_path\"";
        
        // Run the command in the background using popen
        if (substr(PHP_OS, 0, 3) == 'WIN') {
            // Windows: use START command to run in background
            $full_command = "start /B powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File \"$script_path\" 2>&1";
            pclose(popen($full_command, 'r'));
        } else {
            // Linux/Mac (if needed)
            $full_command = "nohup $command > /dev/null 2>&1 &";
            exec($full_command);
        }
        
        // Wait and retry verification multiple times (up to 10 seconds)
        $max_attempts = 5;
        $wait_time = 2; // seconds between attempts
        $service_started = false;
        
        for ($i = 0; $i < $max_attempts; $i++) {
            sleep($wait_time);
            $verify_response = call_flask_service($status_url, [], 'GET');
            
            if ($verify_response['status'] === 'success') {
                $service_started = true;
                break;
            }
        }
        
        if ($service_started) {
            echo json_encode([
                'status' => 'success',
                'message' => 'Flask service started successfully! The ML service is now running.'
            ]);
        } else {
            echo json_encode([
                'status' => 'warning',
                'message' => 'Flask service is starting in the background. Please refresh the status in a moment, or manually run start_service.ps1 from the ml_model folder.'
            ]);
        }
        
    } catch (Exception $e) {
        error_log("Error starting Flask service: " . $e->getMessage());
        echo json_encode([
            'status' => 'error', 
            'message' => 'Failed to start Flask service: ' . $e->getMessage()
        ]);
    }
}

function call_flask_service($url, $data = [], $method = 'POST') {
    try {
        // Use cURL for better reliability with POST requests
        $ch = curl_init($url);
        
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, 120); // 2 minutes for AI processing
        curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 10);
        
        if ($method === 'POST') {
            curl_setopt($ch, CURLOPT_POST, true);
            if (!empty($data)) {
                $json_data = json_encode($data);
                curl_setopt($ch, CURLOPT_POSTFIELDS, $json_data);
                curl_setopt($ch, CURLOPT_HTTPHEADER, [
                    'Content-Type: application/json',
                    'Content-Length: ' . strlen($json_data)
                ]);
            }
        } elseif ($method === 'GET') {
            curl_setopt($ch, CURLOPT_HTTPGET, true);
        }
        
        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $curl_error = curl_error($ch);
        curl_close($ch);
        
        if ($response === false || !empty($curl_error)) {
            error_log("Flask service connection failed: " . $curl_error);
            return ['status' => 'error', 'message' => 'Flask service not available: ' . $curl_error];
        }
        
        if ($http_code !== 200) {
            error_log("Flask service returned HTTP $http_code: " . substr($response, 0, 200));
            return ['status' => 'error', 'message' => "Flask service error (HTTP $http_code)"];
        }
        
        $decoded = json_decode($response, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            error_log("JSON decode error: " . json_last_error_msg() . " Response: " . substr($response, 0, 200));
            return ['status' => 'error', 'message' => 'Invalid JSON response from Flask service'];
        }
        
        return $decoded;
        
    } catch (Exception $e) {
        error_log("Error calling Flask service: " . $e->getMessage());
        return ['status' => 'error', 'message' => 'Flask service error: ' . $e->getMessage()];
    }
}

function get_db_connection() {
    try {
        require_once __DIR__ . '/../includes/conn.php';
        global $conn;
        if (!$conn || $conn->connect_error) {
            throw new Exception("Database connection failed: " . ($conn->connect_error ?? 'Unknown error'));
        }
        return $conn;
    } catch (Exception $e) {
        error_log("Database connection error: " . $e->getMessage());
        throw new Exception("Database connection failed: " . $e->getMessage());
    }
}
?>
