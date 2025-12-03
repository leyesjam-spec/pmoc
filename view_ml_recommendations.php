<?php
/**
 * View AI Recommendations
 * Detailed view of AI-generated counseling recommendations
 */

require_once '../includes/session.php';
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>View Counseling Topics Recommendations - BCPDO System</title>
    <?php include '../includes/header.php'; ?>
</head>

<body class="hold-transition sidebar-mini layout-fixed">
    <div class="wrapper">
        <?php include '../includes/navbar.php'; ?>
        <?php include '../includes/sidebar.php'; ?>
        
        <div class="content-wrapper">
  <section class="content-header">
    <div class="container-fluid">
      <div class="row mb-2">
        <div class="col-sm-6">
          <h1>Counseling Topics Recommendations</h1>
        </div>
      </div>
    </div>
  </section>

  <section class="content">
    <div class="container-fluid">
      <div class="row">
        <div class="col-12">
          <div class="card">
            <div class="card-body">
              <div class="d-flex justify-content-between align-items-center mb-2">
                <h5 id="detailTitle" class="mb-0">Counseling Topics Recommendations</h5>
                <a href="./ml_dashboard.php" class="btn btn-sm btn-secondary">Back to Dashboard</a>
              </div>
              <div id="result" class="mb-2" style="white-space: pre-wrap;"></div>
              <div class="mb-3" id="summaryWrap" style="display:none;">
                <h6 class="text-muted mb-2"><i class="fas fa-lightbulb mr-1"></i>Counseling Topics Analysis Summary</h6>
                <blockquote class="blockquote pl-3 border-left border-info" id="aiSummary"></blockquote>
              </div>
              <div id="cardsWrap"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</div>

        <?php include '../includes/footer.php'; ?>
    </div>

    <?php include '../includes/scripts.php'; ?>

<script>
(function() {
  const result = document.getElementById('result');
  const detailTitle = document.getElementById('detailTitle');
  const aiSummary = document.getElementById('aiSummary');
  const summaryWrap = document.getElementById('summaryWrap');
  const cardsWrap = document.getElementById('cardsWrap');

  function badge(level){
    const map = { High: 'danger', Medium: 'warning', Low: 'success' };
    const cls = map[level] || 'secondary';
    return '<span class="badge badge-' + cls + '">' + level + '</span>';
  }

  async function fetchCouples() {
    const resp = await fetch('../couple_list/get_couples.php', { credentials: 'same-origin' });
    const data = await resp.json();
    if (!data.success) throw new Error('Failed to load couples');
    return data.data;
  }

  async function getAnalysisResults(accessId) {
    const resp = await fetch('./ml_api.php?action=get_analysis&access_id=' + accessId, { credentials: 'same-origin' });
    const data = await resp.json();
    if (data.status !== 'success') throw new Error(data.message || 'Failed to fetch analysis');
    // Check if couple has been analyzed
    if (!data.analyzed) {
      throw new Error('This couple has not been analyzed yet. Please run "Analyze All Couples" from the ML Dashboard first.');
    }
    return data;
  }

  async function renderDetail(accessId) {
    try {
      const couples = await fetchCouples();
      const couple = couples.find(x => String(x.access_id) === String(accessId));
      if (couple) detailTitle.textContent = 'Counseling Topics Recommendations for ' + couple.couple_names;

      const analysis = await getAnalysisResults(accessId);
      
      // Display risk level and summary
      const riskLevel = analysis.risk_level;
      const mlRecommendations = analysis.recommendations || [];
      const focusCategories = analysis.focus_categories || [];
      const mlConfidence = analysis.ml_confidence || 0;
      const riskReasoning = analysis.risk_reasoning || '';
      const counselingReasoning = analysis.counseling_reasoning || '';
      
      console.log('Analysis data:', analysis);
      console.log('ML Recommendations:', mlRecommendations);
      console.log('Focus Categories:', focusCategories);
      console.log('Total categories received:', focusCategories.length);
      console.log('All category names:', focusCategories.map(c => c.name));
      
      // Helper function to generate dynamic analysis text
      const getDynamicAnalysisText = (categories, risk, confidence) => {
        const count = categories.length;
        const criticalCount = categories.filter(c => c.priority === 'Critical').length;
        const highCount = categories.filter(c => c.priority === 'High').length;
        
        let text = '';
        
        // Opening based on count
        if (count === 4) {
          text = 'Identified <strong class="text-primary">all 4 MEAI categories</strong> requiring counseling attention';
        } else if (count === 3) {
          text = 'Identified <strong class="text-primary">3 out of 4 MEAI categories</strong> requiring counseling attention';
        } else if (count === 2) {
          text = 'Identified <strong class="text-primary">2 MEAI categories</strong> requiring counseling attention';
        } else if (count === 1) {
          text = 'Identified <strong class="text-primary">1 MEAI category</strong> requiring counseling attention';
        } else {
          text = 'No priority categories identified - couple appears to be doing well';
        }
        
        // Add priority details if categories exist
        if (count > 0) {
          if (criticalCount > 0) {
            text += `, including <strong class="text-danger">${criticalCount} critical priority area${criticalCount > 1 ? 's' : ''}</strong>`;
          } else if (highCount === count) {
            text += ', all marked as <strong class="text-warning">high priority</strong>';
          } else if (highCount > 0) {
            text += `, with <strong class="text-warning">${highCount} high priority area${highCount > 1 ? 's' : ''}</strong>`;
          }
        }
        
        // Add recommendation based on risk and confidence - consistent with counseling intensity
        if (count > 0) {
          if (risk === 'High' && confidence > 0.6) {
            text += '. <strong>Immediate comprehensive counseling strongly recommended.</strong>';
          } else if (risk === 'High' || confidence > 0.6) {
            text += '. <strong>Comprehensive counseling sessions recommended.</strong>';
          } else if (confidence > 0.3) {
            text += '. <strong>Structured counseling sessions recommended.</strong>';
          } else {
            text += '. <strong>Preventive counseling recommended.</strong>';
          }
        }
        
        return text;
      };
      
      // Use actual personalized recommendations from ML analysis
      const getPersonalizedRecommendations = (mlRecommendations, focusCategories) => {
        const recommendations = [];
        
        // Use the actual personalized recommendations from the ML analysis
        if (mlRecommendations && mlRecommendations.length > 0) {
          // Display the personalized recommendations as-is
          mlRecommendations.forEach(rec => {
            recommendations.push(rec);
          });
        } else {
          // Fallback if no personalized recommendations available
          recommendations.push('⚠️ No personalized recommendations available - please re-analyze this couple');
        }
        
        // Add category-specific explanations
        if (focusCategories && focusCategories.length > 0) {
          // Get all 4 MEAI categories to show which ones are/aren't predicted
          const allCategories = [
            'Marriage And Relationship',
            'Responsible Parenthood', 
            'Planning The Family',
            'Maternal Neonatal Child Health And Nutrition'
          ];
          
          const predictedCategories = focusCategories.map(c => c.name);
          const notPredictedCategories = allCategories.filter(cat => !predictedCategories.includes(cat));
          
          if (notPredictedCategories.length > 0) {
            recommendations.push(`<strong>📋 Category Analysis:</strong> No specific recommendations for ${notPredictedCategories.join(', ')} - these areas show low priority scores (below 20%) indicating the couple is doing well in these areas.`);
          }
        }
        
        return recommendations;
      };
      
      // Create enhanced summary with better readability
      // Calculate priority based on risk level and category scores, not ML confidence
      let priorityPercentage = 0;
      if (riskLevel === 'High') {
        priorityPercentage = 85; // High risk = High priority
      } else if (riskLevel === 'Medium') {
        priorityPercentage = 60; // Medium risk = Medium priority
      } else {
        priorityPercentage = 25; // Low risk = Low priority
      }
      
      // Adjust based on highest category score
      if (focusCategories && focusCategories.length > 0) {
        const maxCategoryScore = Math.max(...focusCategories.map(cat => cat.score));
        const categoryAdjustment = maxCategoryScore * 20; // Scale category score to 0-20%
        priorityPercentage = Math.min(95, priorityPercentage + categoryAdjustment);
      }
      
      const confidencePercentage = priorityPercentage.toFixed(1);
      
      // Get relationship health assessment details with specific reasoning
      let riskIcon = '';
      let riskText = '';
      let riskColor = '';
      
      if (riskLevel === 'High') {
        riskIcon = 'fa-exclamation-circle';
        riskText = 'Significant relationship challenges requiring immediate attention';
        riskColor = 'danger';
      } else if (riskLevel === 'Medium') {
        riskIcon = 'fa-exclamation-triangle';
        riskText = 'Some relationship concerns that need proactive attention';
        riskColor = 'warning';
      } else if (riskLevel === 'Low') {
        riskIcon = 'fa-check-circle';
        riskText = 'Healthy relationship foundation with good communication patterns';
        riskColor = 'success';
      }
      
      // Get counseling recommendation details with specific reasoning
      let confidenceIcon = '';
      let confidenceText = '';
      let confidenceColor = '';
      
      // Set confidence text and color based on priority percentage, not ML confidence
      if (priorityPercentage > 70) {
        confidenceIcon = 'fa-user-md';
        confidenceText = 'Intensive counseling program recommended for comprehensive support';
        confidenceColor = 'danger';
      } else if (priorityPercentage > 40) {
        confidenceIcon = 'fa-handshake';
        confidenceText = 'Structured counseling sessions recommended for relationship development';
        confidenceColor = 'warning';
      } else {
        confidenceIcon = 'fa-heart';
        confidenceText = 'Preventive counseling recommended to maintain relationship health';
        confidenceColor = 'success';
      }
      
      const summary = `
        <div class="row">
          <div class="col-md-6 mb-3">
            <div class="card border-${riskColor}" style="height: 100%;">
              <div class="card-body">
                <h6 class="text-${riskColor} mb-2">
                  <i class="fas ${riskIcon} mr-2"></i>Relationship Assessment
                </h6>
                <div class="mb-2">
                  <span class="badge badge-${riskColor} badge-lg">${riskLevel} Risk</span>
                </div>
                <p class="text-muted mb-2">
                  <i class="fas fa-info-circle mr-1"></i>${riskText}
                </p>
                <small class="text-muted">
                  <i class="fas fa-search mr-1"></i>${riskReasoning || 'Analysis based on couple profile and MEAI assessment data'}
                </small>
              </div>
            </div>
          </div>
          <div class="col-md-6 mb-3">
            <div class="card border-${confidenceColor}" style="height: 100%;">
              <div class="card-body">
                <h6 class="text-${confidenceColor} mb-2">
                  <i class="fas ${confidenceIcon} mr-2"></i>Counseling Recommendation
                </h6>
                <div class="mb-2">
                  <span class="badge badge-${confidenceColor} badge-lg">${confidencePercentage}% Priority</span>
                </div>
                <p class="text-muted mb-2">
                  <i class="fas fa-info-circle mr-1"></i>${confidenceText}
                </p>
                <small class="text-muted">
                  <i class="fas fa-search mr-1"></i>${counselingReasoning || 'Recommendation based on MEAI category analysis and counseling needs assessment'}
                </small>
              </div>
            </div>
          </div>
        </div>
        <div class="alert alert-light border-left-info mb-0">
          <div class="row">
            <div class="col-md-8">
              <i class="fas fa-chart-bar text-primary mr-2"></i>
              <strong>ML Analysis:</strong> ${getDynamicAnalysisText(focusCategories, riskLevel, mlConfidence)}
            </div>
            <div class="col-md-4 text-md-right">
              <small class="text-muted">
                <i class="far fa-clock mr-1"></i>${new Date(analysis.generated_at).toLocaleDateString()}
                <br><i class="fas fa-brain mr-1"></i>${analysis.analysis_method || 'Random Forest ML'}
              </small>
            </div>
          </div>
        </div>
      `;
      
      aiSummary.innerHTML = summary;
      summaryWrap.style.display = '';

      // Display ML recommendations
      cardsWrap.innerHTML = '';
      
      // Focus Categories Card with Enhanced Visuals (FIRST) - Show ALL categories
      if (focusCategories.length > 0) {
        const categoriesCard = document.createElement('div');
        categoriesCard.className = 'mb-4';
        
        // Helper function to get priority badge details (3 levels only)
        const getPriorityBadge = (priority) => {
          const badges = {
            'High': { color: 'danger', icon: 'fa-exclamation-circle', text: 'HIGH' },
            'Moderate': { color: 'warning', icon: 'fa-info-circle', text: 'MODERATE' },
            'Low': { color: 'success', icon: 'fa-check-circle', text: 'LOW' }
          };
          return badges[priority] || badges['Moderate'];
        };
        
        // Helper function to get score color and text (3 levels only)
        const getScoreInfo = (score) => {
          if (score > 0.6) return { color: 'danger', text: 'High Priority Counseling', barClass: 'bg-danger' };
          if (score > 0.3) return { color: 'warning', text: 'Moderate Priority Counseling', barClass: 'bg-warning' };
          return { color: 'success', text: 'Low Priority Monitoring', barClass: 'bg-success' };
        };
        
        categoriesCard.innerHTML = `
          <!-- Score Interpretation Guide (3 Levels Only) -->
          <div class="card bg-light mb-3">
            <div class="card-body p-3">
              <h6 class="mb-2"><i class="fas fa-info-circle mr-2"></i>Score Interpretation Guide</h6>
              <div class="row text-center">
                <div class="col-4">
                  <span class="badge badge-danger badge-lg">60-100%</span>
                  <br><small class="text-muted">High Priority</small>
                </div>
                <div class="col-4">
                  <span class="badge badge-warning badge-lg">30-60%</span>
                  <br><small class="text-muted">Moderate Priority</small>
                </div>
                <div class="col-4">
                  <span class="badge badge-success badge-lg">0-30%</span>
                  <br><small class="text-muted">Low Priority</small>
                </div>
              </div>
            </div>
          </div>
          
          <div class="card border-info">
            <div class="card-header bg-info text-white">
              <h6 class="mb-0">
                <i class="fas fa-bullseye mr-2"></i>Priority Categories for Counseling
              </h6>
            </div>
            <div class="card-body">
              <div class="alert alert-light mb-3">
                <i class="fas fa-lightbulb text-warning mr-2"></i>
                <strong>Counselor Guidance:</strong> All ${focusCategories.length} MEAI categories are displayed below. Focus on categories in order from highest to lowest score. 
                Higher scores indicate stronger need for counseling in that specific MEAI area.
              </div>
              <div class="row">
                ${focusCategories.map((cat, index) => {
                  const priorityBadge = getPriorityBadge(cat.priority);
                  const scoreInfo = getScoreInfo(cat.score);
                  const scorePercentage = (cat.score * 100).toFixed(1);
                  const rankBadge = index === 0 ? '🥇' : (index === 1 ? '🥈' : (index === 2 ? '🥉' : `#${index + 1}`));
                  
                  return `
                  <div class="col-md-6 mb-4">
                    <div class="card border-${scoreInfo.color}">
                      <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                          <h6 class="mb-0">
                            <span class="text-muted mr-2">${rankBadge}</span>
                            ${cat.name}
                          </h6>
                        </div>
                        
                        <div class="mb-2">
                          <span class="badge badge-${priorityBadge.color} mr-2">
                            <i class="fas ${priorityBadge.icon} mr-1"></i>${priorityBadge.text} PRIORITY
                          </span>
                          <span class="badge badge-light">${scorePercentage}% Score</span>
                    </div>
                        
                        <div class="progress mb-2" style="height: 25px;">
                          <div class="progress-bar ${scoreInfo.barClass} progress-bar-striped progress-bar-animated" 
                           role="progressbar" 
                               style="width: ${scorePercentage}%"
                               aria-valuenow="${scorePercentage}" 
                           aria-valuemin="0" 
                           aria-valuemax="100">
                            <strong>${scorePercentage}%</strong>
                          </div>
                        </div>
                        
                        <div class="text-center">
                          <small class="text-${scoreInfo.color}">
                            <i class="fas fa-arrow-right mr-1"></i><strong>${scoreInfo.text} Needed</strong>
                          </small>
                        </div>
                      </div>
                    </div>
                  </div>
                `;
                }).join('')}
              </div>
            </div>
          </div>
        `;
        cardsWrap.appendChild(categoriesCard);
      }
      
      // ML-Driven Recommendations Card (SECOND - BELOW Priority Categories)
      const mainCard = document.createElement('div');
      mainCard.className = 'mb-4';
      
      // Use personalized recommendations from ML analysis
      const finalRecommendations = getPersonalizedRecommendations(mlRecommendations, focusCategories);
      
      mainCard.innerHTML = `
        <div class="card border-primary">
          <div class="card-header bg-primary text-white">
            <h6 class="mb-0"><i class="fas fa-lightbulb mr-2"></i>Counseling Topics Recommendations</h6>
          </div>
          <div class="card-body">
            <ul class="list-unstyled mb-0">
              ${finalRecommendations.map(rec => `<li class="mb-3"><i class="fas fa-arrow-right text-primary mr-2"></i>${rec}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
      cardsWrap.appendChild(mainCard);
      
    } catch (e) {
      result.textContent = 'Error: ' + e.message;
    }
  }

  function formatAIRecommendations(topics) {
    // Format the AI-generated recommendations
    if (typeof topics === 'string') {
      // If it's a raw AI response, try to format it
      const lines = topics.split('\n').filter(line => line.trim());
      let formatted = '<ul class="list-unstyled">';
      
      lines.forEach(line => {
        if (line.trim() && !line.includes('Generate') && !line.includes('Couple profile')) {
          formatted += `<li class="mb-2"><i class="fas fa-arrow-right text-primary mr-2"></i>${line.trim()}</li>`;
        }
      });
      
      formatted += '</ul>';
      return formatted;
    } else if (Array.isArray(topics)) {
      // If it's an array of topics
      let formatted = '<ul class="list-unstyled">';
      topics.forEach(topic => {
        formatted += `<li class="mb-2"><i class="fas fa-arrow-right text-primary mr-2"></i>${topic}</li>`;
      });
      formatted += '</ul>';
      return formatted;
    } else if (topics && topics.ai_generated) {
      // Handle new AI-generated format
      const lines = topics.ai_generated.split('\n').filter(line => line.trim());
      let formatted = '<div class="ai-generated-content">';
      formatted += '<div class="alert alert-info mb-3"><i class="fas fa-robot mr-2"></i>Generated by TinyLlama AI</div>';
      formatted += '<ul class="list-unstyled">';
      
      lines.forEach(line => {
        if (line.trim() && !line.includes('Generate') && !line.includes('Couple profile')) {
          formatted += `<li class="mb-2"><i class="fas fa-arrow-right text-primary mr-2"></i>${line.trim()}</li>`;
        }
      });
      
      formatted += '</ul></div>';
      return formatted;
    } else if (topics && topics.topics) {
      // Handle template-based format
      let formatted = '<ul class="list-unstyled">';
      topics.topics.forEach(topic => {
        formatted += `<li class="mb-2"><i class="fas fa-arrow-right text-primary mr-2"></i>${topic}</li>`;
      });
      formatted += '</ul>';
      if (topics.focus_areas) {
        formatted += `<div class="mt-3"><small class="text-muted"><i class="fas fa-info-circle mr-1"></i>${topics.focus_areas}</small></div>`;
      }
      return formatted;
    } else {
      return '<p class="text-muted">AI recommendations are being processed...</p>';
    }
  }

  const params = new URLSearchParams(window.location.search);
  const accessIdParam = params.get('access_id');
  if (accessIdParam) renderDetail(accessIdParam);
})();
</script>

<style>
.ai-recommendations {
  line-height: 1.6;
}

.ai-recommendations ul li {
  padding: 0.5rem 0;
  border-bottom: 1px solid #f8f9fa;
}

.ai-recommendations ul li:last-child {
  border-bottom: none;
}

.card-header.bg-primary {
  background: linear-gradient(45deg, #007bff, #0056b3) !important;
}

.card-header.bg-info {
  background: linear-gradient(45deg, #17a2b8, #138496) !important;
}

.ai-generated-content .alert {
  border-left: 4px solid #17a2b8;
}

.blockquote {
  border-left: 4px solid #17a2b8;
  padding-left: 1rem;
}

.main-footer {
  margin-top: 20px;
  padding: 15px;
  background-color: #f4f6f9;
  border-top: 1px solid #dee2e6;
}

.content-wrapper {
  min-height: calc(100vh - 200px);
}

/* Enhanced Info Box Styling */
.info-box {
  box-shadow: 0 2px 4px rgba(0,0,0,.08);
  border-radius: 5px;
  transition: transform 0.2s, box-shadow 0.2s;
}

.info-box:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,.15);
}

.info-box-icon {
  border-radius: 5px 0 0 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
}

.info-box-number {
  font-size: 1.5rem;
  font-weight: bold;
}

/* Badge Enhancements */
.badge-lg {
  padding: 0.5rem 0.75rem;
  font-size: 0.9rem;
  font-weight: 600;
}

.badge {
  font-weight: 600;
}

/* Priority Category Card Enhancements */
.card {
  transition: transform 0.2s, box-shadow 0.2s;
}

.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,.1);
}

.progress {
  box-shadow: inset 0 1px 2px rgba(0,0,0,.1);
  border-radius: 5px;
}

.progress-bar {
  font-weight: bold;
  font-size: 0.9rem;
}

/* Alert Styling */
.alert-light {
  background-color: #f8f9fa;
  border-left: 4px solid #ffc107;
}

.alert-info {
  border-left: 4px solid #17a2b8;
}

.border-left-info {
  border-left: 4px solid #17a2b8 !important;
}

/* Summary Card Enhancements */
.card.border-danger,
.card.border-warning,
.card.border-success,
.card.border-info {
  border-width: 2px;
}

.badge-lg {
  font-size: 1.2rem;
  padding: 0.6rem 1rem;
}

/* Responsive Adjustments */
@media (max-width: 768px) {
  .info-box-number {
    font-size: 1.2rem;
  }
  
  .info-box-icon {
    font-size: 1.5rem;
  }
  
  .badge-lg {
    padding: 0.4rem 0.6rem;
    font-size: 0.8rem;
  }
}
</style>
</body>
</html>
