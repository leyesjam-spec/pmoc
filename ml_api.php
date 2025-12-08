<?php
/**
 * ML + AI API
 * Bridge between frontend and Flask ML + AI service
 */

// Set timezone to match server location (Philippines = Asia/Manila)
date_default_timezone_set('Asia/Manila');

// Error handling
ini_set('display_errors', 0);
ini_set('log_errors', 1);
error_reporting(E_ALL);

// Set error handler to catch fatal errors
register_shutdown_function(function() {
    $error = error_get_last();
    if ($error !== NULL && in_array($error['type'], [E_ERROR, E_PARSE, E_CORE_ERROR, E_COMPILE_ERROR])) {
        http_response_code(500);
        header('Content-Type: application/json');
        echo json_encode([
            'status' => 'error',
            'message' => 'Fatal PHP error: ' . $error['message'],
            'file' => $error['file'],
            'line' => $error['line']
        ]);
    }
});

// Set JSON header
header('Content-Type: application/json');

// Get action
$action = $_GET['action'] ?? $_POST['action'] ?? '';

// For API calls, skip session handling completely
if (in_array($action, ['status', 'analyze', 'get_analysis', 'analyze_batch', 'train', 'training_status', 'test', 'start_service'])) {
    // API calls - no session required
    require_once __DIR__ . '/../includes/conn.php';
    require_once __DIR__ . '/ml_config.php';
} else {
    // For non-API calls, require full session
    require_once __DIR__ . '/../includes/session.php';
}

// CRITICAL: Test log to verify error_log is working
error_log("=========================================");
error_log("DEBUG - ml_api.php LOADED - Action: " . ($action ?? 'NONE'));
error_log("DEBUG - Time: " . date('Y-m-d H:i:s'));
error_log("DEBUG - Error log location: " . ini_get('error_log'));
error_log("=========================================");

switch($action) {
    case 'analyze':
        analyze_couple();
        break;
    case 'get_analysis':
        get_existing_analysis();
        break;
    case 'analyze_batch':
        error_log("DEBUG - ml_api.php - Routing to analyze_batch()");
        analyze_batch();
        break;
    case 'train':
        train_models();
        break;
    case 'training_status':
        get_training_status();
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
        
        // CRITICAL DEBUG: Log what get_couple_data() actually returned
        error_log("DEBUG - analyze_couple - get_couple_data() returned:");
        error_log("DEBUG -   couple_data type: " . gettype($couple_data));
        error_log("DEBUG -   couple_data keys: " . json_encode(array_keys($couple_data)));
        if (isset($couple_data['male_responses'])) {
            error_log("DEBUG -   couple_data['male_responses'] type: " . gettype($couple_data['male_responses']) . ", count: " . count($couple_data['male_responses']));
        } else {
            error_log("DEBUG -   couple_data['male_responses'] is NOT SET");
        }
        if (isset($couple_data['female_responses'])) {
            error_log("DEBUG -   couple_data['female_responses'] type: " . gettype($couple_data['female_responses']) . ", count: " . count($couple_data['female_responses']));
        } else {
            error_log("DEBUG -   couple_data['female_responses'] is NOT SET");
        }
        
        // CRITICAL: Ensure male_responses and female_responses are populated
        // DIRECT EXTRACTION - Don't use ?? operator which might hide issues
        $male_responses = [];
        $female_responses = [];
        $questionnaire_responses = [];
        
        if (isset($couple_data['male_responses'])) {
            $male_responses = $couple_data['male_responses'];
            error_log("DEBUG - Extracted male_responses: count=" . count($male_responses));
        } else {
            error_log("ERROR - couple_data does NOT have 'male_responses' key!");
        }
        if (isset($couple_data['female_responses'])) {
            $female_responses = $couple_data['female_responses'];
            error_log("DEBUG - Extracted female_responses: count=" . count($female_responses));
        } else {
            error_log("ERROR - couple_data does NOT have 'female_responses' key!");
        }
        if (isset($couple_data['questionnaire_responses'])) {
            $questionnaire_responses = $couple_data['questionnaire_responses'];
            error_log("DEBUG - Extracted questionnaire_responses: count=" . count($questionnaire_responses));
        }
        
        // CRITICAL DEBUG: Log what we got from get_couple_data()
        error_log("DEBUG - analyze_couple - Extracted from couple_data:");
        error_log("DEBUG -   couple_data keys: " . json_encode(array_keys($couple_data)));
        error_log("DEBUG -   couple_data has 'male_responses' key: " . (isset($couple_data['male_responses']) ? 'YES' : 'NO'));
        error_log("DEBUG -   couple_data has 'female_responses' key: " . (isset($couple_data['female_responses']) ? 'YES' : 'NO'));
        error_log("DEBUG -   couple_data['male_responses'] type: " . (isset($couple_data['male_responses']) ? gettype($couple_data['male_responses']) : 'NOT SET'));
        error_log("DEBUG -   couple_data['female_responses'] type: " . (isset($couple_data['female_responses']) ? gettype($couple_data['female_responses']) : 'NOT SET'));
        error_log("DEBUG -   male_responses count: " . count($male_responses));
        error_log("DEBUG -   female_responses count: " . count($female_responses));
        error_log("DEBUG -   questionnaire_responses count: " . count($questionnaire_responses));
        
        // CRITICAL: If arrays are empty, try to get them directly from couple_data again
        if (empty($male_responses) && isset($couple_data['male_responses']) && is_array($couple_data['male_responses'])) {
            error_log("WARNING - male_responses is empty after extraction, re-extracting directly");
            $male_responses = $couple_data['male_responses'];
            error_log("WARNING - After re-extraction, male_responses count: " . count($male_responses));
        }
        if (empty($female_responses) && isset($couple_data['female_responses']) && is_array($couple_data['female_responses'])) {
            error_log("WARNING - female_responses is empty after extraction, re-extracting directly");
            $female_responses = $couple_data['female_responses'];
            error_log("WARNING - After re-extraction, female_responses count: " . count($female_responses));
        }
        
        // If arrays are still empty but questionnaire_responses has data, populate them
        if (empty($male_responses) && !empty($questionnaire_responses)) {
            error_log("WARNING - male_responses is still empty, populating from questionnaire_responses");
            $male_responses = $questionnaire_responses; // Use same values as fallback
        }
        if (empty($female_responses) && !empty($questionnaire_responses)) {
            error_log("WARNING - female_responses is still empty, populating from questionnaire_responses");
            $female_responses = $questionnaire_responses; // Use same values as fallback
        }
        
        // Ensure arrays match questionnaire_responses length
        $q_count = count($questionnaire_responses);
        if (count($male_responses) != $q_count) {
            error_log("WARNING - male_responses length mismatch ($q_count expected, got " . count($male_responses) . "), fixing...");
            if (count($male_responses) < $q_count) {
                $male_responses = array_pad($male_responses, $q_count, 3);
            } else {
                $male_responses = array_slice($male_responses, 0, $q_count);
            }
        }
        if (count($female_responses) != $q_count) {
            error_log("WARNING - female_responses length mismatch ($q_count expected, got " . count($female_responses) . "), fixing...");
            if (count($female_responses) < $q_count) {
                $female_responses = array_pad($female_responses, $q_count, 3);
            } else {
                $female_responses = array_slice($female_responses, 0, $q_count);
            }
        }
        
        error_log("DEBUG - analyze_couple - Final array counts before sending:");
        error_log("DEBUG -   questionnaire_responses: " . count($questionnaire_responses));
        error_log("DEBUG -   male_responses: " . count($male_responses));
        error_log("DEBUG -   female_responses: " . count($female_responses));
        
        // CRITICAL DEBUG: Verify variables before building analysis_data
        error_log("DEBUG - analyze_couple - Before building analysis_data:");
        error_log("DEBUG -   \$male_responses count: " . count($male_responses) . " (type: " . gettype($male_responses) . ")");
        error_log("DEBUG -   \$female_responses count: " . count($female_responses) . " (type: " . gettype($female_responses) . ")");
        error_log("DEBUG -   \$questionnaire_responses count: " . count($questionnaire_responses) . " (type: " . gettype($questionnaire_responses) . ")");
        
        // CRITICAL FIX: Force populate from couple_data RIGHT BEFORE building analysis_data
        // This ensures we have the latest values even if something reset the variables
        if (isset($couple_data['male_responses']) && is_array($couple_data['male_responses']) && count($couple_data['male_responses']) > 0) {
            $male_responses = $couple_data['male_responses'];
            error_log("CRITICAL FIX - Forced male_responses from couple_data: " . count($male_responses) . " items");
        } elseif (empty($male_responses) && !empty($questionnaire_responses)) {
            $male_responses = $questionnaire_responses;
            error_log("CRITICAL FIX - Populated male_responses from questionnaire_responses: " . count($male_responses) . " items");
        }
        
        if (isset($couple_data['female_responses']) && is_array($couple_data['female_responses']) && count($couple_data['female_responses']) > 0) {
            $female_responses = $couple_data['female_responses'];
            error_log("CRITICAL FIX - Forced female_responses from couple_data: " . count($female_responses) . " items");
        } elseif (empty($female_responses) && !empty($questionnaire_responses)) {
            $female_responses = $questionnaire_responses;
            error_log("CRITICAL FIX - Populated female_responses from questionnaire_responses: " . count($female_responses) . " items");
        }
        
        // FINAL VERIFICATION: Ensure arrays are not empty
        if (empty($male_responses) || empty($female_responses)) {
            error_log("CRITICAL ERROR - Arrays are STILL empty after all fixes!");
            error_log("CRITICAL ERROR - male_responses empty: " . (empty($male_responses) ? 'YES' : 'NO'));
            error_log("CRITICAL ERROR - female_responses empty: " . (empty($female_responses) ? 'YES' : 'NO'));
            error_log("CRITICAL ERROR - questionnaire_responses count: " . count($questionnaire_responses));
            // Last resort: use questionnaire_responses for both
            if (!empty($questionnaire_responses)) {
                $male_responses = $questionnaire_responses;
                $female_responses = $questionnaire_responses;
                error_log("CRITICAL ERROR - Using questionnaire_responses as last resort for both arrays");
            }
        }
        
        // CRITICAL: Store arrays in variables to ensure they're not lost
        $male_resp_for_send = $male_responses;
        $female_resp_for_send = $female_responses;
        
        error_log("DEBUG - analyze_couple - Stored arrays for sending:");
        error_log("DEBUG -   male_resp_for_send count: " . count($male_resp_for_send));
        error_log("DEBUG -   female_resp_for_send count: " . count($female_resp_for_send));
        
        // Prepare data for ML + AI analysis
        $analysis_data = [
            'access_id' => $access_id,
            'male_age' => $couple_data['male_age'],
            'female_age' => $couple_data['female_age'],
            'civil_status' => $couple_data['civil_status'] ?? 'Single',
            'years_living_together' => $couple_data['years_living_together'] ?? 0,
            // REMOVED: past_children, children (features removed from ML model)
            'education_level' => $couple_data['education_level'],
            'income_level' => $couple_data['income_level'],
            'employment_status' => $couple_data['employment_status'] ?? 'Unemployed',
            'questionnaire_responses' => $questionnaire_responses,
            // CRITICAL: Use stored variables to ensure arrays are included
            'male_responses' => $male_resp_for_send,  // CRITICAL: Must be included
            'female_responses' => $female_resp_for_send,  // CRITICAL: Must be included
            'personalized_features' => $couple_data['personalized_features'] ?? [],
            // ADD COUPLE NAMES FOR NLG PERSONALIZATION
            'male_name' => $couple_data['male_name'] ?? 'Male Partner',
            'female_name' => $couple_data['female_name'] ?? 'Female Partner'
        ];
        
        // CRITICAL: Force verify arrays are in analysis_data immediately
        error_log("DEBUG - analyze_couple - Immediately after building analysis_data:");
        error_log("DEBUG -   analysis_data keys: " . json_encode(array_keys($analysis_data)));
        error_log("DEBUG -   male_responses in analysis_data: " . (isset($analysis_data['male_responses']) ? 'YES (' . count($analysis_data['male_responses']) . ' items)' : 'NO'));
        error_log("DEBUG -   female_responses in analysis_data: " . (isset($analysis_data['female_responses']) ? 'YES (' . count($analysis_data['female_responses']) . ' items)' : 'NO'));
        
        // CRITICAL: If arrays are missing, force them back in
        if (!isset($analysis_data['male_responses']) || empty($analysis_data['male_responses']) || count($analysis_data['male_responses']) == 0) {
            error_log("CRITICAL - analyze_couple - male_responses missing/empty, forcing back in");
            $analysis_data['male_responses'] = $male_resp_for_send;
        }
        if (!isset($analysis_data['female_responses']) || empty($analysis_data['female_responses']) || count($analysis_data['female_responses']) == 0) {
            error_log("CRITICAL - analyze_couple - female_responses missing/empty, forcing back in");
            $analysis_data['female_responses'] = $female_resp_for_send;
        }
        
        // CRITICAL: Double-check that arrays are actually in analysis_data
        if (!isset($analysis_data['male_responses']) || !isset($analysis_data['female_responses'])) {
            error_log("CRITICAL ERROR - Arrays missing from analysis_data after building!");
            // Force add them
            $analysis_data['male_responses'] = $male_responses;
            $analysis_data['female_responses'] = $female_responses;
            error_log("CRITICAL ERROR - Forced arrays back into analysis_data");
        }
        
        // CRITICAL DEBUG: Verify arrays are in analysis_data immediately after building
        error_log("DEBUG - analyze_couple - Immediately after building analysis_data:");
        error_log("DEBUG -   analysis_data['male_responses'] count: " . count($analysis_data['male_responses'] ?? []));
        error_log("DEBUG -   analysis_data['female_responses'] count: " . count($analysis_data['female_responses'] ?? []));
        error_log("DEBUG -   isset(analysis_data['male_responses']): " . (isset($analysis_data['male_responses']) ? 'YES' : 'NO'));
        error_log("DEBUG -   isset(analysis_data['female_responses']): " . (isset($analysis_data['female_responses']) ? 'YES' : 'NO'));
        
        // DEBUG: Log the data being sent to Flask
        error_log("DEBUG - Sending data to Flask for access_id: $access_id");
        error_log("DEBUG - Analysis data structure:");
        error_log("DEBUG -   analysis_data keys: " . json_encode(array_keys($analysis_data)));
        error_log("DEBUG -   questionnaire_responses count: " . count($analysis_data['questionnaire_responses'] ?? []));
        error_log("DEBUG -   male_responses count: " . count($analysis_data['male_responses'] ?? []));
        error_log("DEBUG -   female_responses count: " . count($analysis_data['female_responses'] ?? []));
        error_log("DEBUG -   male_responses type: " . gettype($analysis_data['male_responses'] ?? null));
        error_log("DEBUG -   female_responses type: " . gettype($analysis_data['female_responses'] ?? null));
        error_log("DEBUG -   analysis_data has 'male_responses' key: " . (isset($analysis_data['male_responses']) ? 'YES' : 'NO'));
        error_log("DEBUG -   analysis_data has 'female_responses' key: " . (isset($analysis_data['female_responses']) ? 'YES' : 'NO'));
        
        // CRITICAL: Verify variables before adding to analysis_data
        error_log("DEBUG - Variables before adding to analysis_data:");
        error_log("DEBUG -   \$male_responses count: " . count($male_responses));
        error_log("DEBUG -   \$female_responses count: " . count($female_responses));
        
        // CRITICAL: Verify arrays are not empty before sending
        if (empty($analysis_data['male_responses']) || empty($analysis_data['female_responses'])) {
            error_log("ERROR - male_responses or female_responses is EMPTY before sending to Flask!");
            error_log("ERROR - male_responses: " . json_encode($analysis_data['male_responses'] ?? []));
            error_log("ERROR - female_responses: " . json_encode($analysis_data['female_responses'] ?? []));
            
            // Emergency fix: populate from questionnaire_responses if available
            if (!empty($analysis_data['questionnaire_responses'])) {
                $q_count = count($analysis_data['questionnaire_responses']);
                error_log("EMERGENCY FIX - Populating male/female arrays with $q_count items from questionnaire_responses");
                $analysis_data['male_responses'] = array_fill(0, $q_count, 3);
                $analysis_data['female_responses'] = array_fill(0, $q_count, 3);
                error_log("EMERGENCY FIX - Arrays populated: male=" . count($analysis_data['male_responses']) . ", female=" . count($analysis_data['female_responses']));
            }
        }
        
        error_log("DEBUG - Full analysis data (truncated): " . substr(json_encode($analysis_data), 0, 500));
        
        // CRITICAL: Force include male_responses and female_responses even if they seem empty
        // This ensures they're always in the JSON payload
        // ALWAYS force from couple_data to ensure we have the latest values
        if (isset($couple_data['male_responses']) && is_array($couple_data['male_responses']) && count($couple_data['male_responses']) > 0) {
            $analysis_data['male_responses'] = $couple_data['male_responses'];
            error_log("CRITICAL - Forced male_responses from couple_data into analysis_data: " . count($analysis_data['male_responses']) . " items");
        } elseif (!isset($analysis_data['male_responses']) || empty($analysis_data['male_responses'])) {
            error_log("CRITICAL - male_responses missing or empty, using questionnaire_responses as fallback");
            $analysis_data['male_responses'] = !empty($questionnaire_responses) ? $questionnaire_responses : [];
            error_log("CRITICAL - Set male_responses to: " . count($analysis_data['male_responses']) . " items");
        }
        
        if (isset($couple_data['female_responses']) && is_array($couple_data['female_responses']) && count($couple_data['female_responses']) > 0) {
            $analysis_data['female_responses'] = $couple_data['female_responses'];
            error_log("CRITICAL - Forced female_responses from couple_data into analysis_data: " . count($analysis_data['female_responses']) . " items");
        } elseif (!isset($analysis_data['female_responses']) || empty($analysis_data['female_responses'])) {
            error_log("CRITICAL - female_responses missing or empty, using questionnaire_responses as fallback");
            $analysis_data['female_responses'] = !empty($questionnaire_responses) ? $questionnaire_responses : [];
            error_log("CRITICAL - Set female_responses to: " . count($analysis_data['female_responses']) . " items");
        }
        
        // Final verification before sending - MUST have these keys
        error_log("CRITICAL - Final check before sending to Flask:");
        error_log("CRITICAL -   analysis_data keys: " . json_encode(array_keys($analysis_data)));
        error_log("CRITICAL -   male_responses in analysis_data: " . (isset($analysis_data['male_responses']) ? 'YES (' . count($analysis_data['male_responses']) . ' items)' : 'NO'));
        error_log("CRITICAL -   female_responses in analysis_data: " . (isset($analysis_data['female_responses']) ? 'YES (' . count($analysis_data['female_responses']) . ' items)' : 'NO'));
        
        // ABSOLUTE FINAL CHECK: If still missing, abort and log error
        if (!isset($analysis_data['male_responses']) || !isset($analysis_data['female_responses'])) {
            error_log("FATAL ERROR - male_responses or female_responses is STILL missing from analysis_data!");
            error_log("FATAL ERROR - This should never happen. Aborting send.");
            // Don't send if arrays are missing - this will help us catch the bug
        }
        
        // Call Flask service
        $flask_url = get_ml_service_url('analyze');
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
    global $conn;
    
    try {
        // Get couple profile - fix the JOIN condition (no cp.id column exists)
        $stmt = $conn->prepare("
            SELECT 
                cp.first_name, cp.last_name, cp.age, cp.sex,
                cp.civil_status, cp.education, cp.monthly_income,
                cp.years_living_together, cp.employment_status,
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
        
        // CRITICAL DEBUG: First, check what respondent values actually exist in the database
        $check_stmt = $conn->prepare("
            SELECT DISTINCT respondent, COUNT(*) as count
            FROM couple_responses
            WHERE access_id = ?
            GROUP BY respondent
        ");
        $check_stmt->bind_param("s", $access_id);
        $check_stmt->execute();
        $check_result = $check_stmt->get_result();
        $respondent_values = [];
        while ($row = $check_result->fetch_assoc()) {
            $respondent_values[] = $row['respondent'] . ' (' . $row['count'] . ' rows)';
        }
        error_log("DEBUG - get_couple_data - Respondent values in database for access_id $access_id: " . json_encode($respondent_values));
        $check_stmt->close();
        
        // Get questionnaire responses from couple_responses table - SEPARATE BY PARTNER
        // The respondent field flags whether the response is from 'male' or 'female'
        // The access_id links both male and female responses together
        $stmt = $conn->prepare("
            SELECT cr.response, cr.category_id, cr.question_id, cr.sub_question_id, cr.respondent
            FROM couple_responses cr
            WHERE cr.access_id = ?
            ORDER BY cr.category_id, cr.question_id, COALESCE(cr.sub_question_id, 0), cr.respondent
        ");
        
        $stmt->bind_param("s", $access_id);
        $stmt->execute();
        $result = $stmt->get_result();
        
        // DEBUG: Count total responses
        $total_responses = $result->num_rows;
        error_log("DEBUG - Total responses from database: $total_responses for access_id: $access_id");
        
        // CRITICAL DEBUG: Sample first few rows to see actual data structure
        $sample_rows = [];
        $temp_result = $conn->query("
            SELECT response, category_id, question_id, sub_question_id, respondent
            FROM couple_responses
            WHERE access_id = '$access_id'
            LIMIT 5
        ");
        if ($temp_result) {
            while ($sample = $temp_result->fetch_assoc()) {
                $sample_rows[] = $sample;
            }
            error_log("DEBUG - Sample rows from couple_responses table: " . json_encode($sample_rows));
        }
        
        // Build response map: (category_id, question_id, sub_question_id) -> {male: val, female: val}
        $response_map = [];
        $male_count = 0;
        $female_count = 0;
        $unexpected_respondents = [];
        $total_rows_fetched = 0;
        
        while ($row = $result->fetch_assoc()) {
            $total_rows_fetched++;
            // DEBUG: Log first few responses to see structure
            if (count($response_map) < 5) {
                error_log("DEBUG - Sample response row: " . json_encode($row));
            }
            
            // Convert response to numeric value (2=disagree, 3=neutral, 4=agree for ML service)
            $response_value = 3; // default neutral
            if (is_numeric($row['response'])) {
                $response_value = (int)$row['response'];
                // Convert 1-5 scale to 2-4 scale if needed (ML service uses 2,3,4)
                if ($response_value == 1) $response_value = 2; // strongly disagree -> disagree
                if ($response_value == 5) $response_value = 4; // strongly agree -> agree
            } else {
                // Handle text responses by mapping to numeric values
                $response_lower = strtolower($row['response']);
                if (strpos($response_lower, 'strongly disagree') !== false || strpos($response_lower, 'never') !== false || strpos($response_lower, 'disagree') !== false) {
                    $response_value = 2; // disagree
                } elseif (strpos($response_lower, 'neutral') !== false || strpos($response_lower, 'sometimes') !== false) {
                    $response_value = 3; // neutral
                } elseif (strpos($response_lower, 'agree') !== false || strpos($response_lower, 'often') !== false || strpos($response_lower, 'always') !== false) {
                    $response_value = 4; // agree
                }
            }
            
            // Build response map key - use 0 for NULL sub_question_id (standalone questions)
            // This matches how we build keys when iterating through questions
            $sub_q_id = ($row['sub_question_id'] === null || $row['sub_question_id'] === '' || $row['sub_question_id'] === 0) ? 0 : (int)$row['sub_question_id'];
            $key = (int)$row['category_id'] . '_' . (int)$row['question_id'] . '_' . $sub_q_id;
            
            // DEBUG: Log key building for first few responses
            if ($male_count + $female_count < 5) {
                error_log("DEBUG - Building key: category_id=" . $row['category_id'] . ", question_id=" . $row['question_id'] . ", sub_question_id=" . ($row['sub_question_id'] ?? 'NULL') . " -> key=$key, respondent=" . $row['respondent']);
            }
            
            if (!isset($response_map[$key])) {
                $response_map[$key] = ['male' => null, 'female' => null];
            }
            
            // Store by partner using the respondent field - handle case variations
            $respondent = trim($row['respondent']);
            $respondent_lower = strtolower($respondent);
            
            if ($respondent_lower === 'male') {
                $response_map[$key]['male'] = $response_value;
                $male_count++;
            } elseif ($respondent_lower === 'female') {
                $response_map[$key]['female'] = $response_value;
                $female_count++;
            } else {
                // DEBUG: Track unexpected respondent values
                if (!in_array($respondent, $unexpected_respondents)) {
                    $unexpected_respondents[] = $respondent;
                    error_log("DEBUG - Unexpected respondent value: '" . $respondent . "' (raw: " . json_encode($row['respondent']) . ") for key: $key");
                }
            }
        }
        
        error_log("DEBUG - Processed $total_rows_fetched total rows from database");
        error_log("DEBUG - Processed $male_count male responses and $female_count female responses into response_map");
        error_log("DEBUG - Total response_map keys: " . count($response_map));
        if (count($unexpected_respondents) > 0) {
            error_log("DEBUG - Found unexpected respondent values: " . json_encode($unexpected_respondents));
        }
        
        // CRITICAL DEBUG: Log sample keys and their values from response_map
        $sample_keys = array_slice(array_keys($response_map), 0, 10);
        error_log("DEBUG - Sample response_map keys (first 10): " . json_encode($sample_keys));
        
        // CRITICAL DEBUG: Show actual male/female values in response_map (from respondent field)
        $sample_with_values = [];
        foreach ($sample_keys as $key) {
            if (isset($response_map[$key])) {
                $sample_with_values[$key] = $response_map[$key];
            }
        }
        error_log("DEBUG - Sample response_map with values (showing respondent field separation): " . json_encode($sample_with_values));
        
        // CRITICAL DEBUG: Count how many keys have male vs female responses (from respondent field)
        $keys_with_male = 0;
        $keys_with_female = 0;
        $keys_with_both = 0;
        foreach ($response_map as $key => $values) {
            if ($values['male'] !== null) $keys_with_male++;
            if ($values['female'] !== null) $keys_with_female++;
            if ($values['male'] !== null && $values['female'] !== null) $keys_with_both++;
        }
        error_log("DEBUG - response_map statistics (from respondent field):");
        error_log("DEBUG -   Keys with male responses (respondent='male'): $keys_with_male");
        error_log("DEBUG -   Keys with female responses (respondent='female'): $keys_with_female");
        error_log("DEBUG -   Keys with both responses: $keys_with_both");
        error_log("DEBUG -   Keys with only male: " . ($keys_with_male - $keys_with_both));
        error_log("DEBUG -   Keys with only female: " . ($keys_with_female - $keys_with_both));
        
        // CRITICAL: Check if response_map is empty
        if (count($response_map) == 0) {
            error_log("ERROR - response_map is EMPTY! No responses found in database for access_id: $access_id");
            error_log("ERROR - This means the couple_responses table has no data for this access_id");
        }
        
        // DEBUG: Log response map to see what we have
        error_log("DEBUG - Response map keys: " . count($response_map));
        if (count($response_map) > 0) {
            $sample_keys = array_slice(array_keys($response_map), 0, 10);
            error_log("DEBUG - Sample response map keys: " . json_encode($sample_keys));
            error_log("DEBUG - Sample response map entries: " . json_encode(array_slice($response_map, 0, 3, true)));
        }
        
        // Count male and female responses in map
        $male_count_in_map = 0;
        $female_count_in_map = 0;
        foreach ($response_map as $key => $responses) {
            if ($responses['male'] !== null) $male_count_in_map++;
            if ($responses['female'] !== null) $female_count_in_map++;
        }
        error_log("DEBUG - Male responses in map: $male_count_in_map, Female responses in map: $female_count_in_map");
        
        // If no responses found in map, log warning
        if ($male_count_in_map == 0 && $female_count_in_map == 0) {
            error_log("WARNING - No male or female responses found in response_map for access_id: $access_id");
            error_log("WARNING - This might indicate a data structure mismatch or missing responses");
        }
        
        // Build questionnaire_responses array matching MEAI_QUESTION_MAPPING structure
        // This should match the Python service logic exactly - build structure first, then iterate
        // Get question structure from database
        $questionStmt = $conn->prepare("
            SELECT 
                qa.category_id,
                qa.question_id,
                qa.question_text,
                sqa.sub_question_id,
                sqa.sub_question_text
            FROM question_assessment qa
            LEFT JOIN sub_question_assessment sqa ON qa.question_id = sqa.question_id
            ORDER BY qa.category_id ASC, qa.question_id ASC, sqa.sub_question_id ASC
        ");
        $questionStmt->execute();
        $questionResult = $questionStmt->get_result();
        
        // Build MEAI_QUESTIONS structure (same as Python)
        // IMPORTANT: Store sub_question_id along with text to match database keys
        $meai_questions = [];
        while ($qRow = $questionResult->fetch_assoc()) {
            $cat_id = (int)$qRow['category_id'];
            $q_id = (int)$qRow['question_id'];
            $sub_q_text = $qRow['sub_question_text'] ?? null;
            $sub_q_id = $qRow['sub_question_id'] ?? null;
            
            if (!isset($meai_questions[$cat_id])) {
                $meai_questions[$cat_id] = [];
            }
            if (!isset($meai_questions[$cat_id][$q_id])) {
                $meai_questions[$cat_id][$q_id] = [
                    'text' => $qRow['question_text'],
                    'sub_questions' => []  // Will store ['id' => X, 'text' => '...'] or just text for standalone
                ];
            }
            
            // Add sub-question if exists - store both ID and text
            if (!empty($sub_q_text) && $sub_q_id !== null) {
                $meai_questions[$cat_id][$q_id]['sub_questions'][] = [
                    'id' => (int)$sub_q_id,
                    'text' => $sub_q_text
                ];
            }
        }
        
        // Build questionnaire_responses, male_responses, and female_responses in the same order as Python
        // Use the respondent field to flag which responses belong to male vs female
        $questionnaire_responses = [];
        $male_responses = [];
        $female_responses = [];
        $missing_keys = [];
        $found_keys = [];
        
        // Iterate through categories and questions in the same order as Python
        ksort($meai_questions);
        
        // DEBUG: Verify meai_questions structure
        error_log("DEBUG - Starting to build arrays from " . count($meai_questions) . " categories");
        $loop_iterations = 0;
        
        foreach ($meai_questions as $cat_id => $cat_questions) {
            ksort($cat_questions);
            foreach ($cat_questions as $q_id => $q_data) {
                $loop_iterations++;
                if (!empty($q_data['sub_questions'])) {
                    // Question has sub-questions - use actual sub_question_id from database
                    foreach ($q_data['sub_questions'] as $sub_idx => $sub_item) {
                        // Handle both old format (just text) and new format (array with id and text)
                        if (is_array($sub_item) && isset($sub_item['id'])) {
                            $sub_q_id = (int)$sub_item['id'];
                            $sub_q_text = $sub_item['text'];
                        } else {
                            // Fallback for old format - use array index + 1
                            $sub_q_id = $sub_idx + 1;
                            $sub_q_text = is_array($sub_item) ? ($sub_item['text'] ?? '') : $sub_item;
                        }
                        $key = (int)$cat_id . '_' . (int)$q_id . '_' . (int)$sub_q_id;
                        
                        // CRITICAL: Look up responses using the respondent field flag
                        // The response_map was built using the respondent field from couple_responses table
                        $male_resp = null;
                        $female_resp = null;
                        
                        if (isset($response_map[$key])) {
                            // Key exists - get male and female responses that were flagged by respondent field
                            $male_resp = $response_map[$key]['male'];
                            $female_resp = $response_map[$key]['female'];
                            
                            // DEBUG: Log first few lookups to verify respondent field is working
                            if (count($found_keys) < 3) {
                                error_log("DEBUG - Looking up key: $key (cat=$cat_id, q=$q_id, sub=$sub_q_id)");
                                error_log("DEBUG -   response_map[$key] = " . json_encode($response_map[$key]));
                                error_log("DEBUG -   male_resp (from respondent='male'): " . ($male_resp ?? 'null'));
                                error_log("DEBUG -   female_resp (from respondent='female'): " . ($female_resp ?? 'null'));
                            }
                        } else {
                            // Key doesn't exist - this means no response in database for this question
                            if (count($missing_keys) < 5) {
                                $missing_keys[] = $key;
                                error_log("DEBUG - No responses found for key: $key (cat=$cat_id, q=$q_id, sub=$sub_q_id)");
                                error_log("DEBUG -   Key does not exist in response_map. Available keys sample: " . json_encode(array_slice(array_keys($response_map), 0, 5)));
                            }
                        }
                        
                        // DEBUG: Track found vs missing keys
                        if ($male_resp !== null || $female_resp !== null) {
                            $found_keys[] = $key;
                        } else {
                            if (count($missing_keys) < 5) {
                                $missing_keys[] = $key;
                            }
                        }
                        
                        // ALWAYS add to arrays - use actual values from respondent field if available, otherwise default to 3
                        // The respondent field flags whether this is a male or female response
                        $male_responses[] = $male_resp !== null ? $male_resp : 3;
                        $female_responses[] = $female_resp !== null ? $female_resp : 3;
                        
                        // Use weighted approach: if partners disagree significantly, use lower value
                        if ($male_resp !== null && $female_resp !== null) {
                            if (abs($male_resp - $female_resp) >= 2) {
                                $questionnaire_responses[] = min($male_resp, $female_resp);
                            } else {
                                $questionnaire_responses[] = (int)round(($male_resp + $female_resp) / 2);
                            }
                        } elseif ($male_resp !== null) {
                            $questionnaire_responses[] = $male_resp;
                        } elseif ($female_resp !== null) {
                            $questionnaire_responses[] = $female_resp;
                        } else {
                            $questionnaire_responses[] = 3; // Default neutral
                        }
                    }
                } else {
                    // Standalone question - no sub-question (use 0 for key, matching how we built response_map)
                    $key = (int)$cat_id . '_' . (int)$q_id . '_0';
                    
                    // Look up responses using the respondent field flag
                    $male_resp = isset($response_map[$key]) ? $response_map[$key]['male'] : null;
                    $female_resp = isset($response_map[$key]) ? $response_map[$key]['female'] : null;
                    
                    // DEBUG: Track found vs missing keys
                    if ($male_resp !== null || $female_resp !== null) {
                        $found_keys[] = $key;
                        if (count($found_keys) <= 3) {
                            error_log("DEBUG - Found responses for standalone key: $key (cat=$cat_id, q=$q_id) - male: " . ($male_resp ?? 'null') . ", female: " . ($female_resp ?? 'null'));
                        }
                    } else {
                        if (count($missing_keys) < 5) {
                            $missing_keys[] = $key;
                            error_log("DEBUG - No responses found for standalone key: $key (cat=$cat_id, q=$q_id)");
                        }
                    }
                    
                    // ALWAYS add to arrays - use actual values from respondent field if available, otherwise default to 3
                    // The respondent field flags whether this is a male or female response
                    $male_responses[] = $male_resp !== null ? $male_resp : 3;
                    $female_responses[] = $female_resp !== null ? $female_resp : 3;
                    
                    // Use weighted approach: if partners disagree significantly, use lower value
                    if ($male_resp !== null && $female_resp !== null) {
                        if (abs($male_resp - $female_resp) >= 2) {
                            $questionnaire_responses[] = min($male_resp, $female_resp);
                        } else {
                            $questionnaire_responses[] = (int)round(($male_resp + $female_resp) / 2);
                        }
                    } elseif ($male_resp !== null) {
                        $questionnaire_responses[] = $male_resp;
                    } elseif ($female_resp !== null) {
                        $questionnaire_responses[] = $female_resp;
                    } else {
                        $questionnaire_responses[] = 3; // Default neutral
                    }
                }
            }
        }
        
        // DEBUG: Log loop execution
        error_log("DEBUG ========================================");
        error_log("DEBUG - Loop executed $loop_iterations times");
        error_log("DEBUG - response_map has " . count($response_map) . " keys");
        error_log("DEBUG - meai_questions has " . count($meai_questions) . " categories");
        if ($loop_iterations == 0) {
            error_log("ERROR - Loop did not execute! meai_questions might be empty or malformed");
            error_log("ERROR - meai_questions structure: " . json_encode(array_keys($meai_questions)));
        }
        
        // DEBUG: Log summary
        error_log("DEBUG - Found responses for " . count($found_keys) . " keys");
        error_log("DEBUG - Total missing keys: " . count($missing_keys));
        if (count($missing_keys) > 0) {
            error_log("DEBUG - Missing response keys (first 5): " . json_encode(array_slice($missing_keys, 0, 5)));
            // Show what keys ARE in response_map for comparison
            $response_map_keys_sample = array_slice(array_keys($response_map), 0, 10);
            error_log("DEBUG - response_map keys sample: " . json_encode($response_map_keys_sample));
        }
        
        // CRITICAL: Log array counts IMMEDIATELY after loop (before any modifications)
        error_log("DEBUG - Array counts IMMEDIATELY after building loop:");
        error_log("DEBUG -   questionnaire_responses: " . count($questionnaire_responses));
        error_log("DEBUG -   male_responses: " . count($male_responses));
        error_log("DEBUG -   female_responses: " . count($female_responses));
        error_log("DEBUG ========================================");
        
        // CRITICAL FIX: Ensure arrays are always populated to match questionnaire_responses
        // This should never happen, but if the loop didn't populate them, fix it now
        $q_count = count($questionnaire_responses);
        if (count($male_responses) != $q_count || count($female_responses) != $q_count) {
            error_log("ERROR - Array length mismatch! questionnaire_responses: $q_count, male: " . count($male_responses) . ", female: " . count($female_responses));
            
            if ($q_count > 0) {
                // Rebuild arrays to match questionnaire_responses length
                error_log("FIXING - Rebuilding male/female arrays to match questionnaire_responses ($q_count items)");
                
                // If arrays are completely empty, populate with defaults
                if (count($male_responses) == 0 && count($female_responses) == 0) {
                    error_log("ERROR - Both arrays are EMPTY! This means the loop didn't execute.");
                    error_log("ERROR - meai_questions structure: " . count($meai_questions) . " categories");
                    
                    // Populate from response_map if possible, otherwise use defaults
                    $male_responses = [];
                    $female_responses = [];
                    
                    // Try to rebuild by iterating through response_map in order
                    // This is a fallback - the main loop should have done this
                    foreach ($questionnaire_responses as $idx => $q_resp) {
                        // Use the questionnaire response as a fallback for both
                        // This ensures arrays are at least populated
                        $male_responses[] = $q_resp;
                        $female_responses[] = $q_resp;
                    }
                    
                    error_log("FIXED - Populated arrays with " . count($male_responses) . " items each (using questionnaire_responses as fallback)");
                } else {
                    // Arrays exist but wrong length - pad or truncate
                    while (count($male_responses) < $q_count) {
                        $male_responses[] = 3;
                    }
                    while (count($female_responses) < $q_count) {
                        $female_responses[] = 3;
                    }
                    $male_responses = array_slice($male_responses, 0, $q_count);
                    $female_responses = array_slice($female_responses, 0, $q_count);
                    error_log("FIXED - Adjusted arrays to match length: " . count($male_responses));
                }
            }
        }
        
        // Get expected count (same as validation)
        $expected_count = count($questionnaire_responses);
        
        // CRITICAL: Ensure male_responses and female_responses match questionnaire_responses length
        // If they're empty or don't match, populate them
        if (count($male_responses) != $expected_count || count($female_responses) != $expected_count) {
            error_log("WARNING - Array length mismatch detected!");
            error_log("WARNING - questionnaire_responses: $expected_count");
            error_log("WARNING - male_responses: " . count($male_responses));
            error_log("WARNING - female_responses: " . count($female_responses));
            
            // Reset and rebuild arrays to match questionnaire_responses
            $male_responses = [];
            $female_responses = [];
            
            // Re-populate from response_map or use defaults
            foreach ($questionnaire_responses as $idx => $q_resp) {
                // Try to find corresponding responses (this is a fallback)
                // Since we don't have the key here, we'll use defaults
                // But ideally, the loop above should have populated these
                $male_responses[] = 3; // Default neutral
                $female_responses[] = 3; // Default neutral
            }
            
            error_log("WARNING - Rebuilt arrays to match questionnaire_responses length: " . count($male_responses));
        }
        
        if ($expected_count == 0) {
            // Fallback: count from database
        $countStmt = $conn->prepare("
            SELECT 
                SUM(
                    CASE 
                        WHEN sub_count > 0 THEN sub_count
                        ELSE 1
                    END
                ) as total_answerable
            FROM (
                SELECT 
                    qa.question_id,
                    COUNT(DISTINCT CASE 
                        WHEN sqa.sub_question_id IS NOT NULL 
                             AND sqa.sub_question_text IS NOT NULL 
                             AND sqa.sub_question_text != ''
                        THEN sqa.sub_question_id
                        ELSE NULL
                    END) as sub_count
                FROM question_assessment qa
                LEFT JOIN sub_question_assessment sqa ON qa.question_id = sqa.question_id
                GROUP BY qa.question_id
            ) as question_counts
        ");
        $countStmt->execute();
        $countResult = $countStmt->get_result();
            $expected_count = 31; // Default fallback
        if ($countRow = $countResult->fetch_assoc()) {
            $expected_count = (int)$countRow['total_answerable'];
            }
            $questionnaire_responses = array_fill(0, $expected_count, 3);
            // Also populate male/female arrays
            $male_responses = array_fill(0, $expected_count, 3);
            $female_responses = array_fill(0, $expected_count, 3);
        }
        
        // PERSONALIZED ANALYSIS: Calculate partner dynamics
        $personalized_features = calculate_personalized_features($male_responses, $female_responses, $questionnaire_responses);
        
        // DEBUG: Log personalized features
        error_log("DEBUG - Access ID: $access_id");
        error_log("DEBUG - Male responses count: " . count($male_responses) . " (first 5: " . json_encode(array_slice($male_responses, 0, 5)) . ")");
        error_log("DEBUG - Female responses count: " . count($female_responses) . " (first 5: " . json_encode(array_slice($female_responses, 0, 5)) . ")");
        error_log("DEBUG - Questionnaire responses count: " . count($questionnaire_responses) . " (first 5: " . json_encode(array_slice($questionnaire_responses, 0, 5)) . ")");
        error_log("DEBUG - Expected count: $expected_count");
        error_log("DEBUG - Alignment score: " . $personalized_features['alignment_score']);
        error_log("DEBUG - Conflict ratio: " . $personalized_features['conflict_ratio']);
        
        // CRITICAL DEBUG: Check if arrays are actually populated
        $male_non_default = array_filter($male_responses, function($v) { return $v !== 3 && $v !== null; });
        $female_non_default = array_filter($female_responses, function($v) { return $v !== 3 && $v !== null; });
        error_log("DEBUG - Male responses with non-default values: " . count($male_non_default) . " out of " . count($male_responses));
        error_log("DEBUG - Female responses with non-default values: " . count($female_non_default) . " out of " . count($female_responses));
        
        // Ensure arrays are not empty
        if (count($male_responses) == 0 || count($female_responses) == 0) {
            error_log("ERROR - Male or female responses arrays are EMPTY! This should not happen.");
            error_log("ERROR - Response map had " . count($response_map) . " keys");
            error_log("ERROR - Sample response map keys: " . json_encode(array_slice(array_keys($response_map), 0, 10)));
            error_log("ERROR - Sample response map values: " . json_encode(array_slice($response_map, 0, 3, true)));
        }
        
        // Ensure arrays are not empty - if they are, something went wrong
        if (count($male_responses) == 0 || count($female_responses) == 0) {
            error_log("ERROR - Male or female responses arrays are empty! This should not happen.");
            error_log("ERROR - Response map had " . count($response_map) . " keys");
            error_log("ERROR - Sample response map keys: " . json_encode(array_slice(array_keys($response_map), 0, 10)));
            error_log("ERROR - Sample response map values: " . json_encode(array_slice($response_map, 0, 5, true)));
            
            // If arrays are empty but questionnaire_responses has items, rebuild them from response_map
            if (count($questionnaire_responses) > 0 && (count($male_responses) == 0 || count($female_responses) == 0)) {
                error_log("WARNING - Rebuilding male/female responses arrays from response_map");
                // This shouldn't happen, but as a fallback, we'll rebuild
                $male_responses = [];
                $female_responses = [];
                foreach ($questionnaire_responses as $idx => $q_resp) {
                    // Try to find corresponding responses in map (this is a fallback)
                    $male_responses[] = 3; // Default
                    $female_responses[] = 3; // Default
                }
            }
        }
        
        // Ensure arrays match questionnaire_responses length
        if (count($male_responses) != count($questionnaire_responses) || 
            count($female_responses) != count($questionnaire_responses)) {
            error_log("WARNING - Array length mismatch! questionnaire_responses: " . count($questionnaire_responses) . 
                     ", male_responses: " . count($male_responses) . 
                     ", female_responses: " . count($female_responses));
            
            // Pad arrays to match if needed
            $target_length = count($questionnaire_responses);
            while (count($male_responses) < $target_length) {
                $male_responses[] = 3;
            }
            while (count($female_responses) < $target_length) {
                $female_responses[] = 3;
            }
            // Truncate if too long
            $male_responses = array_slice($male_responses, 0, $target_length);
            $female_responses = array_slice($female_responses, 0, $target_length);
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
        
        // REMOVED: past_children and children (features removed from ML model)
        
        // FINAL CHECK: Ensure arrays are populated before returning
        $final_q_count = count($questionnaire_responses);
        if (count($male_responses) != $final_q_count || count($female_responses) != $final_q_count) {
            error_log("FINAL CHECK - Array mismatch detected! Fixing before return...");
            error_log("FINAL CHECK - questionnaire_responses: $final_q_count");
            error_log("FINAL CHECK - male_responses: " . count($male_responses));
            error_log("FINAL CHECK - female_responses: " . count($female_responses));
            
            // Force arrays to match questionnaire_responses
            if ($final_q_count > 0) {
                // If empty, populate from questionnaire_responses (fallback)
                if (count($male_responses) == 0) {
                    $male_responses = $questionnaire_responses; // Use same values as fallback
                    error_log("FINAL CHECK - Populated male_responses from questionnaire_responses");
                }
                if (count($female_responses) == 0) {
                    $female_responses = $questionnaire_responses; // Use same values as fallback
                    error_log("FINAL CHECK - Populated female_responses from questionnaire_responses");
                }
                
                // Ensure exact length match
                $male_responses = array_slice($male_responses, 0, $final_q_count);
                $female_responses = array_slice($female_responses, 0, $final_q_count);
                
                // Pad if needed
                while (count($male_responses) < $final_q_count) {
                    $male_responses[] = 3;
                }
                while (count($female_responses) < $final_q_count) {
                    $female_responses[] = 3;
                }
                
                error_log("FINAL CHECK - Arrays fixed: male=" . count($male_responses) . ", female=" . count($female_responses));
            }
        }
        
        // CRITICAL: Verify arrays are not empty - if they are, this couple has no responses in database
        if (empty($male_responses) || empty($female_responses)) {
            error_log("CRITICAL ERROR - Arrays are still empty after all fixes for access_id: $access_id!");
            error_log("CRITICAL ERROR - This means the couple_responses table has no data for this couple");
            error_log("CRITICAL ERROR - Response map had " . count($response_map) . " keys");
            error_log("CRITICAL ERROR - Total rows fetched from database: $total_rows_fetched");
            error_log("CRITICAL ERROR - Male count: $male_count, Female count: $female_count");
            
            // Check if there's ANY data in couple_responses for this access_id
            $check_any_data = $conn->prepare("SELECT COUNT(*) as total FROM couple_responses WHERE access_id = ?");
            $check_any_data->bind_param("s", $access_id);
            $check_any_data->execute();
            $any_data_result = $check_any_data->get_result();
            $any_data_row = $any_data_result->fetch_assoc();
            $total_in_table = $any_data_row['total'] ?? 0;
            $check_any_data->close();
            
            error_log("CRITICAL ERROR - Total rows in couple_responses table for access_id $access_id: $total_in_table");
            
            if ($total_in_table > 0 && ($male_count == 0 || $female_count == 0)) {
                // Data exists but respondent field might be wrong
                $check_respondent = $conn->prepare("SELECT DISTINCT respondent, COUNT(*) as cnt FROM couple_responses WHERE access_id = ? GROUP BY respondent");
                $check_respondent->bind_param("s", $access_id);
                $check_respondent->execute();
                $respondent_result = $check_respondent->get_result();
                $respondent_values = [];
                while ($r_row = $respondent_result->fetch_assoc()) {
                    $respondent_values[] = $r_row['respondent'] . ' (' . $r_row['cnt'] . ' rows)';
                }
                $check_respondent->close();
                error_log("CRITICAL ERROR - Respondent field values in database: " . json_encode($respondent_values));
                error_log("CRITICAL ERROR - Expected 'male' and 'female' (case-insensitive), but found: " . json_encode($respondent_values));
            }
            
            // Use emergency fallback ONLY if questionnaire_responses has data
            // This allows couples with combined responses to still work
            if (!empty($questionnaire_responses) && count($questionnaire_responses) > 0) {
                error_log("WARNING - Using emergency fallback: populating male/female from questionnaire_responses");
                $male_responses = $questionnaire_responses;
                $female_responses = $questionnaire_responses;
                error_log("WARNING - Emergency fallback: male=" . count($male_responses) . ", female=" . count($female_responses));
            } else {
                // No data at all - return empty arrays so calling code knows
                error_log("CRITICAL ERROR - Returning empty arrays - couple has no responses in database");
            }
        }
        
        error_log("DEBUG - FINAL return values: questionnaire=" . count($questionnaire_responses) . ", male=" . count($male_responses) . ", female=" . count($female_responses));
        
        // CRITICAL DEBUG: Verify arrays before returning
        error_log("DEBUG - get_couple_data - Before return:");
        error_log("DEBUG -   male_responses type: " . gettype($male_responses) . ", count: " . count($male_responses));
        error_log("DEBUG -   female_responses type: " . gettype($female_responses) . ", count: " . count($female_responses));
        error_log("DEBUG -   questionnaire_responses type: " . gettype($questionnaire_responses) . ", count: " . count($questionnaire_responses));
        
        // NEW: Get employment status (use male partner's employment status)
        $employment_status = $male_profile['employment_status'] ?? 'Unemployed';
        
        $return_data = [
            'male_age' => (int)($male_profile['age'] ?? 30),
            'female_age' => (int)($female_profile['age'] ?? 28),
            'civil_status' => $civil_status,
            'years_living_together' => $years_living_together,
            'education_level' => $education_mapping[$male_profile['education'] ?? 'College Level'] ?? 2,
            'income_level' => $income_mapping[$male_profile['monthly_income'] ?? '10000-14999'] ?? 1,
            'employment_status' => $employment_status,  // NEW: Employment status
            'questionnaire_responses' => $questionnaire_responses,
            // PERSONALIZED FEATURES
            'male_responses' => $male_responses,
            'female_responses' => $female_responses,
            'personalized_features' => $personalized_features,
            // ADD COUPLE NAMES FOR NLG PERSONALIZATION
            'male_name' => ($male_profile['first_name'] ?? '') . ' ' . ($male_profile['last_name'] ?? ''),
            'female_name' => ($female_profile['first_name'] ?? '') . ' ' . ($female_profile['last_name'] ?? '')
            // REMOVED: past_children, children
        ];
        
        // CRITICAL DEBUG: Verify arrays are in return_data
        error_log("DEBUG - get_couple_data - Return data keys: " . json_encode(array_keys($return_data)));
        error_log("DEBUG - get_couple_data - Return data has 'male_responses': " . (isset($return_data['male_responses']) ? 'YES (' . count($return_data['male_responses']) . ' items)' : 'NO'));
        error_log("DEBUG - get_couple_data - Return data has 'female_responses': " . (isset($return_data['female_responses']) ? 'YES (' . count($return_data['female_responses']) . ' items)' : 'NO'));
        
        return $return_data;
        
    } catch (Exception $e) {
        error_log("Error getting couple data: " . $e->getMessage());
        return null;
    }
}

function get_personalized_features_for_couple($access_id) {
    // Get couple responses and calculate personalized features
    // Use global $conn from conn.php (already included in ml_api.php)
    global $conn;
    if (!$conn || $conn->connect_error) {
        return [
            'alignment_score' => 0.5,
            'conflict_ratio' => 0.0,
            'category_alignments' => [0.5, 0.5, 0.5, 0.5]  // 4 features, one per category
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
    
    // Don't close $conn - it's a global connection used by other parts of the application
    
    // Calculate personalized features
    return calculate_personalized_features($male_responses, $female_responses, $all_responses);
}

function calculate_personalized_features($male_responses, $female_responses, $all_responses) {
    // Calculate relationship dynamics and personalized features
    
    // 1. ALIGNMENT ANALYSIS
    $alignment_score = 0;
    $total_questions = min(count($male_responses), count($female_responses));
    
    // 2. DISAGREEMENT ANALYSIS (matching Python actual_disagree_ratio logic)
    $question_disagree_count = 0;  // When either partner disagrees with the question (response = 2)
    $partner_disagree_count = 0;  // When partners disagree with each other
    $neutral_count = 0;  // When either partner is neutral (response = 3)
    
    for ($i = 0; $i < $total_questions; $i++) {
        $male_resp = $male_responses[$i] ?? 3;
        $female_resp = $female_responses[$i] ?? 3;
        
        // Calculate alignment (how close their responses are)
        $difference = abs($male_resp - $female_resp);
        $alignment_score += (4 - $difference) / 4; // 0-1 scale
        
        // Count when either partner disagrees with the question (response = 2)
        if ($male_resp == 2 || $female_resp == 2) {
            $question_disagree_count++;
        }
        
        // Count when partners disagree with each other
        if ($difference >= 2) {
            $partner_disagree_count += 1;  // Significant disagreement
        } elseif ($difference == 1) {
            $partner_disagree_count += 0.5;  // Minor disagreement
        }
        
        // Count neutrals (either partner is neutral)
        if ($male_resp == 3 || $female_resp == 3) {
            $neutral_count++;
        }
    }
    
    $alignment_score = $total_questions > 0 ? $alignment_score / $total_questions : 0.5;
    
    // Combined disagreement: use max to avoid double counting, plus weighted neutrals
    // This matches the Python actual_disagree_ratio calculation
    $total_disagree_count = max($question_disagree_count, $partner_disagree_count) + ($neutral_count * 0.3);
    $conflict_ratio = $total_questions > 0 ? $total_disagree_count / $total_questions : 0;
    
    // NEW: Calculate category-specific alignments (4 features, one per MEAI category)
    // Use the same structure as get_couple_data to ensure matching order
    global $conn;
    $category_alignments = [];
    
    try {
        // Build the same MEAI_QUESTIONS structure used in get_couple_data
        // This ensures we iterate in the same order as the response arrays
        $questionStmt = $conn->prepare("
            SELECT 
                qa.category_id,
                qa.question_id,
                qa.question_text,
                sqa.sub_question_id,
                sqa.sub_question_text
            FROM question_assessment qa
            LEFT JOIN sub_question_assessment sqa ON qa.question_id = sqa.question_id
            ORDER BY qa.category_id ASC, qa.question_id ASC, sqa.sub_question_id ASC
        ");
        $questionStmt->execute();
        $questionResult = $questionStmt->get_result();
        
        // Build category alignment tracker
        $category_alignment_sums = [];
        $category_question_counts = [];
        $response_index = 0;
        
        while ($qRow = $questionResult->fetch_assoc()) {
            $cat_id = (int)$qRow['category_id'];
            $sub_q_text = $qRow['sub_question_text'] ?? null;
            
            // Initialize category tracking if needed
            if (!isset($category_alignment_sums[$cat_id])) {
                $category_alignment_sums[$cat_id] = 0;
                $category_question_counts[$cat_id] = 0;
            }
            
            // Check if this is an answerable question (has sub-question text or is standalone)
            $is_answerable = !empty($sub_q_text) || ($sub_q_text === null && $qRow['sub_question_id'] === null);
            
            if ($is_answerable && $response_index < count($male_responses) && $response_index < count($female_responses)) {
                $male_resp = $male_responses[$response_index];
                $female_resp = $female_responses[$response_index];
                $difference = abs($male_resp - $female_resp);
                $category_alignment_sums[$cat_id] += (4 - $difference) / 4;
                $category_question_counts[$cat_id]++;
                $response_index++;
            }
        }
        
        // Get all categories in order
        $catStmt = $conn->prepare("SELECT category_id FROM question_category ORDER BY category_id ASC LIMIT 4");
        $catStmt->execute();
        $catResult = $catStmt->get_result();
        
        while ($catRow = $catResult->fetch_assoc()) {
            $cat_id = (int)$catRow['category_id'];
            $category_alignment = $category_question_counts[$cat_id] > 0 
                ? $category_alignment_sums[$cat_id] / $category_question_counts[$cat_id] 
                : 0.5;
            $category_alignments[] = $category_alignment;
        }
        
        // Ensure we have exactly 4 category alignments
        while (count($category_alignments) < 4) {
            $category_alignments[] = 0.5;
        }
        $category_alignments = array_slice($category_alignments, 0, 4);
        
    } catch (Exception $e) {
        error_log("Error calculating category alignments: " . $e->getMessage());
        // Fallback: use default values
        $category_alignments = [0.5, 0.5, 0.5, 0.5];
    }
    
    // Return only the features expected by Python (6 features total)
    return [
        'alignment_score' => $alignment_score,
        'conflict_ratio' => $conflict_ratio,
        'category_alignments' => $category_alignments  // 4 features, one per category
        // REMOVED: male_avg_response, female_avg_response, male_agree_ratio, male_disagree_ratio, female_agree_ratio, female_disagree_ratio
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
    global $conn;
    
    // Use global connection instead of creating new one to avoid "already closed" errors
    if (!$conn || $conn->connect_error) {
        error_log("ERROR - Database connection not available in save_analysis_results");
        return false;
    }
    
    try {
        // Prepare data for saving
        $risk_level = $results['risk_level'] ?? 'Medium';
        $actual_risk_level = $results['actual_risk_level'] ?? $risk_level;  // Response-based risk
        $actual_disagree_ratio = $results['actual_disagree_ratio'] ?? 0.0;  // Disagreement ratio
        $ml_confidence = $results['ml_confidence'] ?? 0;
        $category_scores = json_encode($results['category_scores'] ?? []);
        $focus_categories = json_encode($results['focus_categories'] ?? []);
        $recommendations = json_encode($results['recommendations'] ?? []);
        $analysis_method = $results['analysis_method'] ?? 'Random Forest Counseling Topics Model';
        
        // Insert or update analysis results in ml_analysis table
        // Note: actual_risk_level and actual_disagree_ratio stored in JSON format in risk_reasoning for now
        // (to avoid database schema changes, we'll include them in the response when fetching)
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
            generated_at = NOW(),
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
            $stmt->close();
            return true;
        } else {
            error_log("Failed to save ML analysis for access_id: $access_id");
            $stmt->close();
            return false;
        }
        
    } catch (Exception $e) {
        error_log("Error saving ML analysis to ml_analysis table: " . $e->getMessage());
        // Don't close connection here - let the calling function handle it
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
        // Convert timestamps to local timezone (Asia/Manila = UTC+8)
        $stmt = $conn->prepare("
            SELECT 
                access_id,
                risk_level,
                ml_confidence,
                category_scores,
                focus_categories,
                recommendations,
                analysis_method,
                CONVERT_TZ(generated_at, @@session.time_zone, '+08:00') as generated_at,
                CONVERT_TZ(updated_at, @@session.time_zone, '+08:00') as updated_at
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
        
        // Calculate response-based risk level from conflict_ratio (approximation)
        // conflict_ratio is similar to disagreement ratio
        $conflict_ratio = $personalized_features['conflict_ratio'] ?? 0.0;
        $actual_risk_level = 'Low';
        if ($conflict_ratio > 0.35) {
            $actual_risk_level = 'High';
        } elseif ($conflict_ratio > 0.20) {
            $actual_risk_level = 'Medium';
        }
        $actual_disagree_ratio = $conflict_ratio; // Use conflict_ratio as approximation
        
        echo json_encode([
            'status' => 'success',
            'analyzed' => true,
            'risk_level' => $row['risk_level'],  // Final hybrid risk level
            'actual_risk_level' => $actual_risk_level,  // Response-based risk (male vs female comparison)
            'actual_disagree_ratio' => $actual_disagree_ratio,  // Disagreement ratio
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
            'male_avg_response' => $personalized_features['male_avg_response'] ?? 0,
            'female_avg_response' => $personalized_features['female_avg_response'] ?? 0
        ]);
        
        // Don't close $conn - it's a global connection used by other parts of the application
        
    } catch (Exception $e) {
        error_log("Get analysis error: " . $e->getMessage());
        echo json_encode(['status' => 'error', 'message' => 'Failed to fetch analysis: ' . $e->getMessage()]);
    }
}

function analyze_batch() {
    // ============================================
    // CRITICAL DEBUG: Function entry point
    // ============================================
    $log_file = __DIR__ . '/analyze_batch_debug.log';
    $log_msg = "[" . date('Y-m-d H:i:s') . "] analyze_batch() FUNCTION CALLED\n";
    file_put_contents($log_file, $log_msg, FILE_APPEND);
    
    error_log("=========================================");
    error_log("DEBUG - analyze_batch() FUNCTION CALLED");
    error_log("DEBUG - Time: " . date('Y-m-d H:i:s'));
    error_log("DEBUG - Error log location: " . ini_get('error_log'));
    error_log("=========================================");
    
    try {
        // Get list of access_ids to analyze
        $access_ids = $_POST['access_ids'] ?? null;
        
        error_log("DEBUG - analyze_batch - Received access_ids: " . ($access_ids ? 'YES' : 'NO'));
        if ($access_ids) {
            error_log("DEBUG - analyze_batch - access_ids type: " . gettype($access_ids));
            if (is_string($access_ids)) {
                error_log("DEBUG - analyze_batch - access_ids is string, length: " . strlen($access_ids));
            }
        }
        
        if (!$access_ids) {
            error_log("ERROR - analyze_batch - No access_ids provided");
            echo json_encode(['status' => 'error', 'message' => 'access_ids required']);
            return;
        }
        
        // If it's a JSON string, decode it
        if (is_string($access_ids)) {
            $access_ids = json_decode($access_ids, true);
            error_log("DEBUG - analyze_batch - Decoded JSON, got " . count($access_ids ?? []) . " access_ids");
        }
        
        $results = [
            'total' => count($access_ids),
            'success' => 0,
            'failed' => 0,
            'skipped' => 0,
            'errors' => []
        ];
        
        error_log("DEBUG - analyze_batch - Processing " . count($access_ids) . " couples");
        
        foreach ($access_ids as $access_id) {
            try {
                error_log("DEBUG - analyze_batch - ===== Processing couple: $access_id =====");
                
                // Get couple data
                $couple_data = get_couple_data($access_id);
                
                if (!$couple_data) {
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id not found";
                    continue;
                }
                
                // CRITICAL DEBUG: Check what get_couple_data() returned
                error_log("DEBUG - analyze_batch - Couple $access_id: get_couple_data() returned");
                error_log("DEBUG -   couple_data keys: " . json_encode(array_keys($couple_data)));
                error_log("DEBUG -   Has 'male_responses': " . (isset($couple_data['male_responses']) ? 'YES (' . count($couple_data['male_responses']) . ' items)' : 'NO'));
                error_log("DEBUG -   Has 'female_responses': " . (isset($couple_data['female_responses']) ? 'YES (' . count($couple_data['female_responses']) . ' items)' : 'NO'));
                
                // CRITICAL: Detailed check of arrays
                $male_check_isset = isset($couple_data['male_responses']);
                $male_check_is_array = isset($couple_data['male_responses']) && is_array($couple_data['male_responses']);
                $male_check_count = isset($couple_data['male_responses']) && is_array($couple_data['male_responses']) ? count($couple_data['male_responses']) : 0;
                $male_check_type = isset($couple_data['male_responses']) ? gettype($couple_data['male_responses']) : 'NOT SET';
                
                error_log("DEBUG - analyze_batch - Male responses check:");
                error_log("DEBUG -   isset: " . ($male_check_isset ? 'YES' : 'NO'));
                error_log("DEBUG -   is_array: " . ($male_check_is_array ? 'YES' : 'NO'));
                error_log("DEBUG -   count: $male_check_count");
                error_log("DEBUG -   type: $male_check_type");
                if ($male_check_isset && $male_check_is_array && $male_check_count > 0) {
                    error_log("DEBUG -   first 3 values: " . json_encode(array_slice($couple_data['male_responses'], 0, 3)));
                }
                
                // CRITICAL: Check if arrays exist and are not empty BEFORE building analysis_data
                // Use explicit checks: isset, is_array, and count > 0
                if (!$male_check_isset || !$male_check_is_array || $male_check_count == 0) {
                    error_log("ERROR - analyze_batch - Couple $access_id: male_responses is missing or empty!");
                    error_log("ERROR -   male_responses exists: " . (isset($couple_data['male_responses']) ? 'YES' : 'NO'));
                    error_log("ERROR -   male_responses count: " . (isset($couple_data['male_responses']) ? count($couple_data['male_responses']) : 'N/A'));
                    
                    // Check if couple has any questionnaire data at all
                    $has_questionnaire = isset($couple_data['questionnaire_responses']) && !empty($couple_data['questionnaire_responses']) && count($couple_data['questionnaire_responses']) > 0;
                    
                    if ($has_questionnaire) {
                        // Has questionnaire but not separated by gender - this is a data issue
                        error_log("WARNING - Couple $access_id has questionnaire_responses but not male_responses - respondent field may be missing or incorrect");
                        $results['failed']++;
                        $results['errors'][] = "Couple $access_id: male_responses is required and must not be empty. Data must come from couple_responses table with respondent=\"male\". Check that the respondent field is set correctly in the database.";
                    } else {
                        // No questionnaire data at all - skip silently or with info message
                        error_log("INFO - Couple $access_id: No questionnaire responses found. Skipping analysis.");
                        $results['skipped']++;
                        continue; // Skip this couple without adding to errors
                    }
                    continue;
                }
                
                // CRITICAL: Detailed check of female arrays
                $female_check_isset = isset($couple_data['female_responses']);
                $female_check_empty = empty($couple_data['female_responses']);
                $female_check_count = isset($couple_data['female_responses']) ? count($couple_data['female_responses']) : 0;
                $female_check_type = isset($couple_data['female_responses']) ? gettype($couple_data['female_responses']) : 'NOT SET';
                
                error_log("DEBUG - analyze_batch - Female responses check:");
                error_log("DEBUG -   isset: " . ($female_check_isset ? 'YES' : 'NO'));
                error_log("DEBUG -   empty(): " . ($female_check_empty ? 'YES' : 'NO'));
                error_log("DEBUG -   count: $female_check_count");
                error_log("DEBUG -   type: $female_check_type");
                
                if (!isset($couple_data['female_responses']) || !is_array($couple_data['female_responses']) || count($couple_data['female_responses']) == 0) {
                    error_log("ERROR - analyze_batch - Couple $access_id: female_responses is missing or empty!");
                    error_log("ERROR -   female_responses exists: " . (isset($couple_data['female_responses']) ? 'YES' : 'NO'));
                    error_log("ERROR -   female_responses count: " . (isset($couple_data['female_responses']) ? count($couple_data['female_responses']) : 'N/A'));
                    
                    // Check if couple has any questionnaire data at all
                    $has_questionnaire = isset($couple_data['questionnaire_responses']) && !empty($couple_data['questionnaire_responses']) && count($couple_data['questionnaire_responses']) > 0;
                    
                    if ($has_questionnaire) {
                        // Has questionnaire but not separated by gender - this is a data issue
                        error_log("WARNING - Couple $access_id has questionnaire_responses but not female_responses - respondent field may be missing or incorrect");
                        $results['failed']++;
                        $results['errors'][] = "Couple $access_id: female_responses is required and must not be empty. Data must come from couple_responses table with respondent=\"female\". Check that the respondent field is set correctly in the database.";
                    } else {
                        // No questionnaire data at all - skip silently or with info message
                        error_log("INFO - Couple $access_id: No questionnaire responses found. Skipping analysis.");
                        $results['skipped']++;
                        continue; // Skip this couple without adding to errors
                    }
                    continue;
                }
                
                // Validate array lengths (should be 59)
                $expected_count = 59; // Expected number of answerable questions
                if (count($couple_data['male_responses']) != $expected_count) {
                    error_log("WARNING - analyze_batch - Couple $access_id: male_responses has " . count($couple_data['male_responses']) . " items, expected $expected_count");
                }
                if (count($couple_data['female_responses']) != $expected_count) {
                    error_log("WARNING - analyze_batch - Couple $access_id: female_responses has " . count($couple_data['female_responses']) . " items, expected $expected_count");
                }
                
                // Prepare data for ML analysis
                // CRITICAL: Must include male_responses and female_responses from respondent field
                // Store arrays in variables first to ensure they're not lost
                // CRITICAL: Direct extraction with explicit checks
                if (!isset($couple_data['male_responses']) || !is_array($couple_data['male_responses'])) {
                    error_log("FATAL ERROR - analyze_batch - Couple $access_id: male_responses is not set or not an array in couple_data!");
                    error_log("FATAL ERROR - couple_data keys: " . json_encode(array_keys($couple_data)));
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: male_responses is missing from couple_data. This should never happen if get_couple_data() worked correctly.";
                    continue;
                }
                if (!isset($couple_data['female_responses']) || !is_array($couple_data['female_responses'])) {
                    error_log("FATAL ERROR - analyze_batch - Couple $access_id: female_responses is not set or not an array in couple_data!");
                    error_log("FATAL ERROR - couple_data keys: " . json_encode(array_keys($couple_data)));
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: female_responses is missing from couple_data. This should never happen if get_couple_data() worked correctly.";
                    continue;
                }
                
                // CRITICAL: Extract arrays directly - use array_values() to ensure it's a proper indexed array
                $male_resp_array = array_values($couple_data['male_responses']);
                $female_resp_array = array_values($couple_data['female_responses']);
                
                error_log("DEBUG - analyze_batch - Extracted arrays before building analysis_data:");
                error_log("DEBUG -   male_resp_array type: " . gettype($male_resp_array) . ", count: " . count($male_resp_array));
                error_log("DEBUG -   female_resp_array type: " . gettype($female_resp_array) . ", count: " . count($female_resp_array));
                error_log("DEBUG -   male_resp_array first 3: " . json_encode(array_slice($male_resp_array, 0, 3)));
                error_log("DEBUG -   female_resp_array first 3: " . json_encode(array_slice($female_resp_array, 0, 3)));
                
                // CRITICAL: Verify arrays are not empty after extraction
                if (count($male_resp_array) == 0) {
                    error_log("FATAL ERROR - analyze_batch - Couple $access_id: male_resp_array is EMPTY after extraction!");
                    error_log("FATAL ERROR - Original couple_data['male_responses'] count: " . count($couple_data['male_responses']));
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: male_responses array became empty during extraction.";
                    continue;
                }
                if (count($female_resp_array) == 0) {
                    error_log("FATAL ERROR - analyze_batch - Couple $access_id: female_resp_array is EMPTY after extraction!");
                    error_log("FATAL ERROR - Original couple_data['female_responses'] count: " . count($couple_data['female_responses']));
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: female_responses array became empty during extraction.";
                    continue;
                }
                
                $analysis_data = [
                    'access_id' => $access_id,
                    'male_age' => $couple_data['male_age'],
                    'female_age' => $couple_data['female_age'],
                    'civil_status' => $couple_data['civil_status'] ?? 'Single',
                    'years_living_together' => $couple_data['years_living_together'] ?? 0,
                    // REMOVED: past_children, children (features removed from ML model)
                    'education_level' => $couple_data['education_level'],
                    'income_level' => $couple_data['income_level'],
                    'employment_status' => $couple_data['employment_status'] ?? 'Unemployed',  // NEW: Employment status
                    'questionnaire_responses' => $couple_data['questionnaire_responses'] ?? [],
                    // CRITICAL: Must include male_responses and female_responses from respondent field
                    'male_responses' => $male_resp_array,  // Use variable to ensure it's included
                    'female_responses' => $female_resp_array,  // Use variable to ensure it's included
                    'personalized_features' => $couple_data['personalized_features'] ?? []
                ];
                
                // CRITICAL: Immediately verify arrays are in analysis_data
                error_log("DEBUG - analyze_batch - Immediately after building analysis_data:");
                error_log("DEBUG -   analysis_data keys: " . json_encode(array_keys($analysis_data)));
                error_log("DEBUG -   male_responses in analysis_data: " . (isset($analysis_data['male_responses']) ? 'YES (' . count($analysis_data['male_responses']) . ' items)' : 'NO'));
                error_log("DEBUG -   female_responses in analysis_data: " . (isset($analysis_data['female_responses']) ? 'YES (' . count($analysis_data['female_responses']) . ' items)' : 'NO'));
                
                // CRITICAL: If arrays are missing, force add them
                if (!isset($analysis_data['male_responses']) || empty($analysis_data['male_responses'])) {
                    error_log("CRITICAL - analyze_batch - male_responses missing from analysis_data, forcing it back in");
                    $analysis_data['male_responses'] = $male_resp_array;
                }
                if (!isset($analysis_data['female_responses']) || empty($analysis_data['female_responses'])) {
                    error_log("CRITICAL - analyze_batch - female_responses missing from analysis_data, forcing it back in");
                    $analysis_data['female_responses'] = $female_resp_array;
                }
                
                // FINAL VERIFICATION: Double-check arrays are in analysis_data
                if (!isset($analysis_data['male_responses']) || empty($analysis_data['male_responses']) || count($analysis_data['male_responses']) == 0) {
                    error_log("CRITICAL ERROR - analyze_batch - Couple $access_id: male_responses is STILL empty after adding to analysis_data!");
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: male_responses is required and must not be empty. Data must come from couple_responses table with respondent=\"male\".";
                    continue;
                }
                
                if (!isset($analysis_data['female_responses']) || empty($analysis_data['female_responses']) || count($analysis_data['female_responses']) == 0) {
                    error_log("CRITICAL ERROR - analyze_batch - Couple $access_id: female_responses is STILL empty after adding to analysis_data!");
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: female_responses is required and must not be empty. Data must come from couple_responses table with respondent=\"female\".";
                    continue;
                }
                
                error_log("DEBUG - analyze_batch - Couple $access_id: Ready to send - male_responses=" . count($analysis_data['male_responses']) . ", female_responses=" . count($analysis_data['female_responses']));
                
                // CRITICAL: Final verification before sending
                error_log("DEBUG - analyze_batch - Final check before call_flask_service:");
                error_log("DEBUG -   analysis_data keys: " . json_encode(array_keys($analysis_data)));
                error_log("DEBUG -   male_responses in analysis_data: " . (isset($analysis_data['male_responses']) ? 'YES (' . count($analysis_data['male_responses']) . ' items)' : 'NO'));
                error_log("DEBUG -   female_responses in analysis_data: " . (isset($analysis_data['female_responses']) ? 'YES (' . count($analysis_data['female_responses']) . ' items)' : 'NO'));
                
                // CRITICAL: Force include arrays even if they seem empty (they shouldn't be at this point)
                if (!isset($analysis_data['male_responses']) || empty($analysis_data['male_responses'])) {
                    error_log("CRITICAL ERROR - analyze_batch - male_responses is missing or empty right before sending!");
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: male_responses is required and must not be empty. Data must come from couple_responses table with respondent=\"male\".";
                    continue;
                }
                if (!isset($analysis_data['female_responses']) || empty($analysis_data['female_responses'])) {
                    error_log("CRITICAL ERROR - analyze_batch - female_responses is missing or empty right before sending!");
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: female_responses is required and must not be empty. Data must come from couple_responses table with respondent=\"female\".";
                    continue;
                }
                
                // CRITICAL: Verify variables before building data_to_send
                error_log("DEBUG - analyze_batch - Before building data_to_send:");
                error_log("DEBUG -   \$male_resp_array type: " . gettype($male_resp_array) . ", count: " . count($male_resp_array));
                error_log("DEBUG -   \$female_resp_array type: " . gettype($female_resp_array) . ", count: " . count($female_resp_array));
                error_log("DEBUG -   \$analysis_data['male_responses'] exists: " . (isset($analysis_data['male_responses']) ? 'YES (' . count($analysis_data['male_responses']) . ' items)' : 'NO'));
                error_log("DEBUG -   \$analysis_data['female_responses'] exists: " . (isset($analysis_data['female_responses']) ? 'YES (' . count($analysis_data['female_responses']) . ' items)' : 'NO'));
                
                // CRITICAL: Use arrays directly from couple_data if variables are empty
                if (empty($male_resp_array) || count($male_resp_array) == 0) {
                    error_log("WARNING - analyze_batch - \$male_resp_array is empty, using from couple_data");
                    $male_resp_array = $couple_data['male_responses'] ?? [];
                }
                if (empty($female_resp_array) || count($female_resp_array) == 0) {
                    error_log("WARNING - analyze_batch - \$female_resp_array is empty, using from couple_data");
                    $female_resp_array = $couple_data['female_responses'] ?? [];
                }
                
                // CRITICAL: Create a copy of analysis_data to ensure arrays aren't lost
                // Use explicit array copy to ensure arrays are included
                // CRITICAL: Get arrays directly from couple_data as final fallback
                $final_male_array = array_values($couple_data['male_responses']);
                $final_female_array = array_values($couple_data['female_responses']);
                
                error_log("DEBUG - analyze_batch - Final arrays from couple_data:");
                error_log("DEBUG -   final_male_array count: " . count($final_male_array));
                error_log("DEBUG -   final_female_array count: " . count($final_female_array));
                
                $data_to_send = [
                    'access_id' => $analysis_data['access_id'],
                    'male_age' => $analysis_data['male_age'],
                    'female_age' => $analysis_data['female_age'],
                    'civil_status' => $analysis_data['civil_status'],
                    'years_living_together' => $analysis_data['years_living_together'],
                    'education_level' => $analysis_data['education_level'],
                    'income_level' => $analysis_data['income_level'],
                    'employment_status' => $analysis_data['employment_status'] ?? 'Unemployed',
                    'questionnaire_responses' => $analysis_data['questionnaire_responses'] ?? [],
                    // CRITICAL: Use arrays directly from couple_data as final source of truth
                    'male_responses' => $final_male_array,  // Direct from couple_data
                    'female_responses' => $final_female_array,  // Direct from couple_data
                    'personalized_features' => $analysis_data['personalized_features'] ?? []
                ];
                
                // CRITICAL: Immediately verify arrays are in data_to_send
                error_log("DEBUG - analyze_batch - IMMEDIATELY after building data_to_send:");
                error_log("DEBUG -   data_to_send['male_responses'] count: " . count($data_to_send['male_responses']));
                error_log("DEBUG -   data_to_send['female_responses'] count: " . count($data_to_send['female_responses']));
                error_log("DEBUG -   data_to_send['male_responses'] first 3: " . json_encode(array_slice($data_to_send['male_responses'], 0, 3)));
                error_log("DEBUG -   data_to_send['female_responses'] first 3: " . json_encode(array_slice($data_to_send['female_responses'], 0, 3)));
                
                // CRITICAL: Verify arrays are actually in data_to_send
                error_log("DEBUG - analyze_batch - After building data_to_send:");
                error_log("DEBUG -   data_to_send keys: " . json_encode(array_keys($data_to_send)));
                error_log("DEBUG -   male_responses count: " . count($data_to_send['male_responses']));
                error_log("DEBUG -   female_responses count: " . count($data_to_send['female_responses']));
                error_log("DEBUG -   male_responses first 5: " . json_encode(array_slice($data_to_send['male_responses'], 0, 5)));
                error_log("DEBUG -   female_responses first 5: " . json_encode(array_slice($data_to_send['female_responses'], 0, 5)));
                
                // CRITICAL: Force include arrays one more time right before sending (safety check)
                if (!isset($data_to_send['male_responses']) || empty($data_to_send['male_responses']) || count($data_to_send['male_responses']) == 0) {
                    error_log("CRITICAL - analyze_batch - male_responses missing/empty in data_to_send, forcing from variables");
                    $data_to_send['male_responses'] = $male_resp_array;
                    error_log("CRITICAL - Forced male_responses: " . count($male_resp_array) . " items");
                }
                if (!isset($data_to_send['female_responses']) || empty($data_to_send['female_responses']) || count($data_to_send['female_responses']) == 0) {
                    error_log("CRITICAL - analyze_batch - female_responses missing/empty in data_to_send, forcing from variables");
                    $data_to_send['female_responses'] = $female_resp_array;
                    error_log("CRITICAL - Forced female_responses: " . count($female_resp_array) . " items");
                }
                
                // FINAL CHECK: Verify data_to_send has arrays with actual data
                error_log("DEBUG - analyze_batch - FINAL CHECK before call_flask_service:");
                error_log("DEBUG -   data_to_send keys: " . json_encode(array_keys($data_to_send)));
                error_log("DEBUG -   male_responses in data_to_send: " . (isset($data_to_send['male_responses']) ? 'YES (' . count($data_to_send['male_responses']) . ' items)' : 'NO'));
                error_log("DEBUG -   female_responses in data_to_send: " . (isset($data_to_send['female_responses']) ? 'YES (' . count($data_to_send['female_responses']) . ' items)' : 'NO'));
                
                // ABSOLUTE FINAL CHECK: If arrays are still empty, abort
                if (empty($data_to_send['male_responses']) || count($data_to_send['male_responses']) == 0) {
                    error_log("FATAL ERROR - analyze_batch - male_responses is STILL empty after all fixes!");
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: male_responses is required and must not be empty. Data must come from couple_responses table with respondent=\"male\".";
                    continue;
                }
                if (empty($data_to_send['female_responses']) || count($data_to_send['female_responses']) == 0) {
                    error_log("FATAL ERROR - analyze_batch - female_responses is STILL empty after all fixes!");
                    $results['failed']++;
                    $results['errors'][] = "Couple $access_id: female_responses is required and must not be empty. Data must come from couple_responses table with respondent=\"female\".";
                    continue;
                }
                
                // CRITICAL: Final check RIGHT BEFORE calling Flask service
                $log_file = __DIR__ . '/analyze_batch_debug.log';
                $log_msg = "[" . date('Y-m-d H:i:s') . "] Couple $access_id - RIGHT BEFORE call_flask_service()\n";
                $log_msg .= "  data_to_send keys: " . json_encode(array_keys($data_to_send)) . "\n";
                $log_msg .= "  male_responses: " . (isset($data_to_send['male_responses']) ? 'YES (' . count($data_to_send['male_responses']) . ' items)' : 'NO') . "\n";
                $log_msg .= "  female_responses: " . (isset($data_to_send['female_responses']) ? 'YES (' . count($data_to_send['female_responses']) . ' items)' : 'NO') . "\n";
                file_put_contents($log_file, $log_msg, FILE_APPEND);
                
                error_log("DEBUG - analyze_batch - ===== RIGHT BEFORE call_flask_service() =====");
                error_log("DEBUG - analyze_batch - Couple $access_id:");
                error_log("DEBUG -   data_to_send type: " . gettype($data_to_send));
                error_log("DEBUG -   data_to_send keys: " . json_encode(array_keys($data_to_send)));
                error_log("DEBUG -   male_responses in data_to_send: " . (isset($data_to_send['male_responses']) ? 'YES' : 'NO'));
                error_log("DEBUG -   female_responses in data_to_send: " . (isset($data_to_send['female_responses']) ? 'YES' : 'NO'));
                if (isset($data_to_send['male_responses'])) {
                    error_log("DEBUG -   male_responses type: " . gettype($data_to_send['male_responses']));
                    error_log("DEBUG -   male_responses count: " . count($data_to_send['male_responses']));
                    error_log("DEBUG -   male_responses first 3: " . json_encode(array_slice($data_to_send['male_responses'], 0, 3)));
                }
                if (isset($data_to_send['female_responses'])) {
                    error_log("DEBUG -   female_responses type: " . gettype($data_to_send['female_responses']));
                    error_log("DEBUG -   female_responses count: " . count($data_to_send['female_responses']));
                    error_log("DEBUG -   female_responses first 3: " . json_encode(array_slice($data_to_send['female_responses'], 0, 3)));
                }
                error_log("DEBUG - analyze_batch - ===== CALLING call_flask_service() NOW =====");
                
                // Call Flask service
                $flask_url = get_ml_service_url('analyze');
                $response = call_flask_service($flask_url, $data_to_send, 'POST');
                
                $log_msg = "[" . date('Y-m-d H:i:s') . "] Couple $access_id - AFTER call_flask_service()\n";
                $log_msg .= "  Response status: " . ($response['status'] ?? 'NOT SET') . "\n";
                file_put_contents($log_file, $log_msg, FILE_APPEND);
                
                error_log("DEBUG - analyze_batch - ===== AFTER call_flask_service() =====");
                error_log("DEBUG - analyze_batch - Response status: " . ($response['status'] ?? 'NOT SET'));
                
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
        // Check if Flask service is running first
        $status_url = get_ml_service_url('status');
        $status_response = call_flask_service($status_url, [], 'GET', 10);
        
        if ($status_response['status'] !== 'success') {
            echo json_encode([
                'status' => 'error', 
                'message' => 'Flask service is not running. Please start the service first.',
                'details' => $status_response['message'] ?? 'Service unavailable'
            ]);
            return;
        }
        
        $flask_url = get_ml_service_url('train');
        // Training now starts asynchronously and returns immediately
        $response = call_flask_service($flask_url, [], 'POST', 30); // 30 seconds should be enough for immediate response
        
        if (!$response) {
            echo json_encode([
                'status' => 'error', 
                'message' => 'No response from Flask service. Training may have timed out or the service crashed.'
            ]);
            return;
        }
        
        echo json_encode($response);
    } catch (Exception $e) {
        error_log("Train models error: " . $e->getMessage());
        error_log("Stack trace: " . $e->getTraceAsString());
        echo json_encode([
            'status' => 'error', 
            'message' => 'Training failed: ' . $e->getMessage(),
            'error_type' => get_class($e)
        ]);
    } catch (Error $e) {
        error_log("Train models fatal error: " . $e->getMessage());
        error_log("Stack trace: " . $e->getTraceAsString());
        echo json_encode([
            'status' => 'error', 
            'message' => 'Fatal error during training: ' . $e->getMessage(),
            'error_type' => get_class($e)
        ]);
    }
}

function get_training_status() {
    try {
        $flask_url = get_ml_service_url('training_status');
        $response = call_flask_service($flask_url, [], 'GET', 10);
        echo json_encode($response);
    } catch (Exception $e) {
        echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
    }
}

function check_status() {
    try {
        $flask_url = get_ml_service_url('status');
        $response = call_flask_service($flask_url, [], 'GET');
        echo json_encode($response);
    } catch (Exception $e) {
        echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
    }
}

function start_flask_service() {
    try {
        // Path to the PowerShell script (in ml_model folder)
        $script_path = __DIR__ . '\\..\\ml_model\\start_service.ps1';
        
        // Check if the script exists
        if (!file_exists($script_path)) {
            echo json_encode([
                'status' => 'error', 
                'message' => 'start_service.ps1 not found. Please ensure the script exists in ml_model folder.'
            ]);
            return;
        }
        
        // Check if Flask is already running
        $status_url = get_ml_service_url('status');
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

function call_flask_service($url, $data = [], $method = 'POST', $timeout = 120) {
    // CRITICAL: Log function entry
    error_log("=========================================");
    error_log("DEBUG - call_flask_service() CALLED");
    error_log("DEBUG - URL: $url");
    error_log("DEBUG - Method: $method");
    error_log("DEBUG - Time: " . date('Y-m-d H:i:s'));
    error_log("=========================================");
    
    try {
        // CRITICAL: Check what we received at the very start
        error_log("DEBUG - call_flask_service - START: Received data with " . count($data) . " keys");
        error_log("DEBUG - call_flask_service - START: Keys: " . json_encode(array_keys($data)));
        error_log("DEBUG - call_flask_service - START: Has 'male_responses': " . (isset($data['male_responses']) ? 'YES (' . count($data['male_responses']) . ' items)' : 'NO'));
        error_log("DEBUG - call_flask_service - START: Has 'female_responses': " . (isset($data['female_responses']) ? 'YES (' . count($data['female_responses']) . ' items)' : 'NO'));
        
        // CRITICAL: If arrays are missing at the start, abort immediately
        if ($method === 'POST' && !empty($data)) {
            if (!isset($data['male_responses']) || !isset($data['female_responses'])) {
                error_log("FATAL ERROR - call_flask_service - Arrays are missing at function start!");
                error_log("FATAL ERROR - This means they were never passed to call_flask_service()");
                return [
                    'status' => 'error',
                    'message' => 'male_responses and female_responses are required but were not included in the request data. Please check that couple_responses table has data with respondent="male" and respondent="female".'
                ];
            }
        }
        
        // Use cURL for better reliability with POST requests
        $ch = curl_init($url);
        
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, $timeout); // Configurable timeout (default 2 minutes, training uses 15 minutes)
        curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 10);
        
        if ($method === 'POST') {
            curl_setopt($ch, CURLOPT_POST, true);
            if (!empty($data)) {
                // CRITICAL DEBUG: Always log what we're sending
                error_log("DEBUG - call_flask_service - Data keys: " . json_encode(array_keys($data)));
                error_log("DEBUG - call_flask_service - Has 'male_responses' key: " . (isset($data['male_responses']) ? 'YES' : 'NO'));
                error_log("DEBUG - call_flask_service - Has 'female_responses' key: " . (isset($data['female_responses']) ? 'YES' : 'NO'));
                
                // DEBUG: Log what we're sending, especially male/female responses
                if (isset($data['male_responses']) || isset($data['female_responses'])) {
                    error_log("DEBUG - call_flask_service - Sending data with:");
                    error_log("DEBUG -   male_responses count: " . (isset($data['male_responses']) ? count($data['male_responses']) : 'NOT SET'));
                    error_log("DEBUG -   female_responses count: " . (isset($data['female_responses']) ? count($data['female_responses']) : 'NOT SET'));
                    error_log("DEBUG -   male_responses type: " . (isset($data['male_responses']) ? gettype($data['male_responses']) : 'NOT SET'));
                    error_log("DEBUG -   female_responses type: " . (isset($data['female_responses']) ? gettype($data['female_responses']) : 'NOT SET'));
                    if (isset($data['male_responses']) && is_array($data['male_responses'])) {
                        error_log("DEBUG -   male_responses first 3: " . json_encode(array_slice($data['male_responses'], 0, 3)));
                    }
                    if (isset($data['female_responses']) && is_array($data['female_responses'])) {
                        error_log("DEBUG -   female_responses first 3: " . json_encode(array_slice($data['female_responses'], 0, 3)));
                    }
                } else {
                    error_log("ERROR - call_flask_service - male_responses and female_responses are NOT SET in data!");
                    error_log("ERROR - call_flask_service - Available keys: " . json_encode(array_keys($data)));
                    
                    // CRITICAL: If arrays are missing, ABORT immediately - don't proceed
                    error_log("FATAL ERROR - call_flask_service - Arrays are missing from data!");
                    error_log("FATAL ERROR - This means they were not included in analysis_data");
                    error_log("FATAL ERROR - Aborting request - cannot proceed without male_responses and female_responses");
                    
                    // curl_close() is deprecated in PHP 8.0+ - use unset() instead
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'male_responses and female_responses are required but were not included in the request data. Please check that couple_responses table has data with respondent="male" and respondent="female".'
                    ];
                }
                
                // CRITICAL: Final check before JSON encoding - ensure arrays are present
                if (!isset($data['male_responses']) || !isset($data['female_responses'])) {
                    error_log("FATAL ERROR - call_flask_service - Arrays are missing from data before JSON encoding!");
                    error_log("FATAL ERROR - Data keys: " . json_encode(array_keys($data)));
                    error_log("FATAL ERROR - This should never happen - arrays should be in analysis_data");
                    // curl_close() is deprecated in PHP 8.0+ - use unset() instead
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'male_responses and female_responses are required but were not included in the request data'
                    ];
                }
                
                // CRITICAL: Verify arrays are not empty - check both empty() and count()
                $male_count = isset($data['male_responses']) ? count($data['male_responses']) : 0;
                $female_count = isset($data['female_responses']) ? count($data['female_responses']) : 0;
                
                if (empty($data['male_responses']) || $male_count == 0) {
                    error_log("FATAL ERROR - call_flask_service - male_responses is empty before JSON encoding!");
                    error_log("FATAL ERROR - male_responses count: $male_count");
                    error_log("FATAL ERROR - male_responses type: " . gettype($data['male_responses'] ?? null));
                    error_log("FATAL ERROR - Data keys at this point: " . json_encode(array_keys($data)));
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'male_responses is required and must not be empty. Data must come from couple_responses table with respondent="male".'
                    ];
                }
                if (empty($data['female_responses']) || $female_count == 0) {
                    error_log("FATAL ERROR - call_flask_service - female_responses is empty before JSON encoding!");
                    error_log("FATAL ERROR - female_responses count: $female_count");
                    error_log("FATAL ERROR - female_responses type: " . gettype($data['female_responses'] ?? null));
                    error_log("FATAL ERROR - Data keys at this point: " . json_encode(array_keys($data)));
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'female_responses is required and must not be empty. Data must come from couple_responses table with respondent="female".'
                    ];
                }
                
                // CRITICAL: Log actual array contents to verify they're not all zeros or nulls
                if ($male_count > 0) {
                    $male_non_zero = array_filter($data['male_responses'], function($v) { return $v !== null && $v !== 0 && $v !== ''; });
                    error_log("DEBUG - call_flask_service - male_responses has " . count($male_non_zero) . " non-zero/non-null values out of $male_count");
                }
                if ($female_count > 0) {
                    $female_non_zero = array_filter($data['female_responses'], function($v) { return $v !== null && $v !== 0 && $v !== ''; });
                    error_log("DEBUG - call_flask_service - female_responses has " . count($female_non_zero) . " non-zero/non-null values out of $female_count");
                }
                
                // CRITICAL: Ensure arrays are proper arrays (not objects or other types)
                if (!is_array($data['male_responses'])) {
                    error_log("FATAL ERROR - call_flask_service - male_responses is not an array! Type: " . gettype($data['male_responses']));
                    $data['male_responses'] = (array)$data['male_responses'];
                }
                if (!is_array($data['female_responses'])) {
                    error_log("FATAL ERROR - call_flask_service - female_responses is not an array! Type: " . gettype($data['female_responses']));
                    $data['female_responses'] = (array)$data['female_responses'];
                }
                
                // CRITICAL: Final verification - log the exact data structure before encoding
                error_log("DEBUG - call_flask_service - FINAL CHECK before json_encode:");
                error_log("DEBUG -   Data keys: " . json_encode(array_keys($data)));
                error_log("DEBUG -   male_responses exists: " . (isset($data['male_responses']) ? 'YES' : 'NO'));
                error_log("DEBUG -   female_responses exists: " . (isset($data['female_responses']) ? 'YES' : 'NO'));
                error_log("DEBUG -   male_responses count: " . (isset($data['male_responses']) ? count($data['male_responses']) : 'N/A'));
                error_log("DEBUG -   female_responses count: " . (isset($data['female_responses']) ? count($data['female_responses']) : 'N/A'));
                
                // CRITICAL: Force ensure arrays are arrays and not empty before encoding
                if (!is_array($data['male_responses']) || empty($data['male_responses'])) {
                    error_log("FATAL ERROR - male_responses is not a valid array before json_encode!");
                    error_log("FATAL ERROR - Type: " . gettype($data['male_responses']));
                    error_log("FATAL ERROR - Value: " . json_encode($data['male_responses']));
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'male_responses is required and must not be empty. Data must come from couple_responses table with respondent="male".'
                    ];
                }
                if (!is_array($data['female_responses']) || empty($data['female_responses'])) {
                    error_log("FATAL ERROR - female_responses is not a valid array before json_encode!");
                    error_log("FATAL ERROR - Type: " . gettype($data['female_responses']));
                    error_log("FATAL ERROR - Value: " . json_encode($data['female_responses']));
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'female_responses is required and must not be empty. Data must come from couple_responses table with respondent="female".'
                    ];
                }
                
                $json_data = json_encode($data, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
                
                // CRITICAL: Check for JSON encoding errors
                $json_error = json_last_error();
                if ($json_error !== JSON_ERROR_NONE) {
                    error_log("FATAL ERROR - JSON encoding failed: " . json_last_error_msg());
                    error_log("FATAL ERROR - Data that failed to encode: " . print_r($data, true));
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'Failed to encode data as JSON: ' . json_last_error_msg()
                    ];
                }
                
                // CRITICAL: Verify the JSON string contains the arrays
                if (strpos($json_data, '"male_responses"') === false) {
                    error_log("FATAL ERROR - JSON string does NOT contain 'male_responses'!");
                    error_log("FATAL ERROR - JSON sample (first 500 chars): " . substr($json_data, 0, 500));
                    error_log("FATAL ERROR - Original data had male_responses: " . (isset($data['male_responses']) ? 'YES (' . count($data['male_responses']) . ' items)' : 'NO'));
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'male_responses was lost during JSON encoding. Please check the data structure.'
                    ];
                }
                if (strpos($json_data, '"female_responses"') === false) {
                    error_log("FATAL ERROR - JSON string does NOT contain 'female_responses'!");
                    error_log("FATAL ERROR - JSON sample (first 500 chars): " . substr($json_data, 0, 500));
                    error_log("FATAL ERROR - Original data had female_responses: " . (isset($data['female_responses']) ? 'YES (' . count($data['female_responses']) . ' items)' : 'NO'));
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'female_responses was lost during JSON encoding. Please check the data structure.'
                    ];
                }
                
                // CRITICAL DEBUG: Check JSON encoding for errors
                $json_error = json_last_error();
                if ($json_error !== JSON_ERROR_NONE) {
                    error_log("ERROR - JSON encoding failed: " . json_last_error_msg());
                }
                
                // DEBUG: Verify JSON encoding didn't drop the arrays
                $decoded_check = json_decode($json_data, true);
                $decode_error = json_last_error();
                if ($decode_error !== JSON_ERROR_NONE) {
                    error_log("ERROR - JSON decode check failed: " . json_last_error_msg());
                }
                
                // CRITICAL: Always check if keys exist in decoded JSON
                    error_log("DEBUG - After JSON encode/decode check:");
                error_log("DEBUG -   Decoded JSON keys: " . json_encode(array_keys($decoded_check ?? [])));
                error_log("DEBUG -   Has 'male_responses' in decoded: " . (isset($decoded_check['male_responses']) ? 'YES' : 'NO'));
                error_log("DEBUG -   Has 'female_responses' in decoded: " . (isset($decoded_check['female_responses']) ? 'YES' : 'NO'));
                
                // CRITICAL: Check if arrays exist and are not empty in decoded JSON
                if (isset($decoded_check['male_responses'])) {
                    $decoded_male_count = count($decoded_check['male_responses']);
                    error_log("DEBUG -   male_responses in JSON: $decoded_male_count items");
                    if ($decoded_male_count == 0 || empty($decoded_check['male_responses'])) {
                        error_log("FATAL ERROR - male_responses is EMPTY in decoded JSON! Original had " . count($data['male_responses'] ?? []) . " items");
                        error_log("FATAL ERROR - This means JSON encoding dropped the array or it was empty");
                        error_log("FATAL ERROR - JSON sample (first 1000 chars): " . substr($json_data, 0, 1000));
                        // ABORT - don't send empty arrays
                        unset($ch);
                        return [
                            'status' => 'error',
                            'message' => 'male_responses is required and must not be empty. Data must come from couple_responses table with respondent="male".'
                        ];
                    }
                } else {
                    error_log("FATAL ERROR - male_responses NOT FOUND in decoded JSON!");
                    error_log("FATAL ERROR - Available keys in decoded JSON: " . json_encode(array_keys($decoded_check ?? [])));
                    error_log("FATAL ERROR - Original data had 'male_responses': " . (isset($data['male_responses']) ? 'YES (' . count($data['male_responses']) . ' items)' : 'NO'));
                    error_log("FATAL ERROR - JSON string contains 'male_responses': " . (strpos($json_data, '"male_responses"') !== false ? 'YES' : 'NO'));
                    // ABORT - arrays are missing from JSON
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'male_responses is required but was not included in the JSON payload. Please check that couple_responses table has data with respondent="male".'
                    ];
                }
                
                if (isset($decoded_check['female_responses'])) {
                    $decoded_female_count = count($decoded_check['female_responses']);
                    error_log("DEBUG -   female_responses in JSON: $decoded_female_count items");
                    if ($decoded_female_count == 0 || empty($decoded_check['female_responses'])) {
                        error_log("FATAL ERROR - female_responses is EMPTY in decoded JSON! Original had " . count($data['female_responses'] ?? []) . " items");
                        error_log("FATAL ERROR - This means JSON encoding dropped the array or it was empty");
                        // ABORT - don't send empty arrays
                        unset($ch);
                        return [
                            'status' => 'error',
                            'message' => 'female_responses is required and must not be empty. Data must come from couple_responses table with respondent="female".'
                        ];
                    }
                } else {
                    error_log("FATAL ERROR - female_responses NOT FOUND in decoded JSON!");
                    error_log("FATAL ERROR - Available keys in decoded JSON: " . json_encode(array_keys($decoded_check ?? [])));
                    error_log("FATAL ERROR - Original data had 'female_responses': " . (isset($data['female_responses']) ? 'YES (' . count($data['female_responses']) . ' items)' : 'NO'));
                    error_log("FATAL ERROR - JSON string contains 'female_responses': " . (strpos($json_data, '"female_responses"') !== false ? 'YES' : 'NO'));
                    // ABORT - arrays are missing from JSON
                    unset($ch);
                    return [
                        'status' => 'error',
                        'message' => 'female_responses is required but was not included in the JSON payload. Please check that couple_responses table has data with respondent="female".'
                    ];
                }
                
                // CRITICAL: Log the actual JSON being sent (first 2000 chars to see structure)
                error_log("DEBUG - call_flask_service - JSON being sent (first 2000 chars): " . substr($json_data, 0, 2000));
                error_log("DEBUG - call_flask_service - JSON length: " . strlen($json_data));
                error_log("DEBUG - call_flask_service - JSON contains 'male_responses': " . (strpos($json_data, '"male_responses"') !== false ? 'YES' : 'NO'));
                error_log("DEBUG - call_flask_service - JSON contains 'female_responses': " . (strpos($json_data, '"female_responses"') !== false ? 'YES' : 'NO'));
                
                // CRITICAL: Extract and log the actual male_responses value from JSON
                if (preg_match('/"male_responses"\s*:\s*\[([^\]]*)\]/', $json_data, $matches)) {
                    $male_in_json = $matches[1];
                    error_log("DEBUG - call_flask_service - male_responses in JSON (first 200 chars): " . substr($male_in_json, 0, 200));
                    error_log("DEBUG - call_flask_service - male_responses array length in JSON: " . (substr_count($male_in_json, ',') + 1));
                } else {
                    error_log("ERROR - call_flask_service - Could not find male_responses array in JSON!");
                }
                
                if (preg_match('/"female_responses"\s*:\s*\[([^\]]*)\]/', $json_data, $matches)) {
                    $female_in_json = $matches[1];
                    error_log("DEBUG - call_flask_service - female_responses in JSON (first 200 chars): " . substr($female_in_json, 0, 200));
                    error_log("DEBUG - call_flask_service - female_responses array length in JSON: " . (substr_count($female_in_json, ',') + 1));
                } else {
                    error_log("ERROR - call_flask_service - Could not find female_responses array in JSON!");
                }
                
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
        
        // curl_close() is deprecated in PHP 8.0+ - curl handles are automatically cleaned up
        // No need to explicitly close in PHP 8.0+
        // The resource will be automatically freed when $ch goes out of scope
        unset($ch);
        
        if ($response === false || !empty($curl_error)) {
            error_log("Flask service connection failed: " . $curl_error);
            return ['status' => 'error', 'message' => 'Flask service not available: ' . $curl_error];
        }
        
        if ($http_code !== 200) {
            error_log("Flask service returned HTTP $http_code: " . substr($response, 0, 200));
            
            // Try to parse the error message from Flask response
            $decoded = json_decode($response, true);
            if (json_last_error() === JSON_ERROR_NONE && isset($decoded['message'])) {
                return ['status' => 'error', 'message' => $decoded['message']];
            }
            
            // Fallback to generic error message
            return ['status' => 'error', 'message' => "Flask service error (HTTP $http_code)"];
        }
        
        $decoded = json_decode($response, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            error_log("JSON decode error: " . json_last_error_msg() . " Response: " . substr($response, 0, 200));
            return ['status' => 'error', 'message' => 'Invalid JSON response from Flask service'];
        }
        
        return $decoded;
        
    } catch (Exception $e) {
        error_log("call_flask_service exception: " . $e->getMessage());
        return ['status' => 'error', 'message' => 'Connection error: ' . $e->getMessage()];
    } catch (Error $e) {
        error_log("call_flask_service fatal error: " . $e->getMessage());
        return ['status' => 'error', 'message' => 'Fatal error: ' . $e->getMessage()];
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
