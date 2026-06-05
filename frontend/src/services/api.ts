import axios from 'axios'

const API_BASE_URL = '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export interface PredictionRequest {
  player_id: string
  course_id: string
}

export interface OutcomeProbabilities {
  make_cut: number
  top_10: number
  win: number
}

export interface PredictionResponse {
  player_id: string
  course_id: string
  fit_score?: number
  outcome_probabilities: OutcomeProbabilities
  confidence: number
  top_features: Array<{ feature: string; importance: number }>
}

export interface RankingRequest {
  tournament_name: string
  courses: string[]
  aggregation_method?: string
  top_n?: number
}

export interface ExplanationRequest {
  player_id: string
  course_id: string
  top_n?: number
}

// API Methods
export const predictPlayerOutcome = async (
  playerID: string,
  courseID: string
): Promise<PredictionResponse> => {
  const response = await api.post('/predictions/player-outcome', {
    player_id: playerID,
    course_id: courseID,
  })
  return response.data
}

export const getTournamentRankings = async (
  tournamentName: string,
  courses: string[],
  topN: number = 50
) => {
  const response = await api.post('/rankings/tournament', {
    tournament_name: tournamentName,
    courses,
    top_n: topN,
  })
  return response.data
}

export const getPlayerCourseFits = async (playerID: string) => {
  const response = await api.get(`/rankings/player/${playerID}/course-fits`)
  return response.data
}

export const getExplanation = async (
  playerID: string,
  courseID: string,
  topN: number = 10
) => {
  const response = await api.post('/explanations/local', {
    player_id: playerID,
    course_id: courseID,
    top_n: topN,
  })
  return response.data
}

export const getFeatureImportance = async (topN: number = 15) => {
  const response = await api.get('/explanations/feature-importance', {
    params: { top_n: topN },
  })
  return response.data
}

export const updatePlayerStats = async () => {
  const response = await api.post('/data/update-stats')
  return response.data
}

export const getDataInfo = async () => {
  const response = await api.get('/data/info')
  return response.data
}

export const getHealth = async () => {
  const response = await api.get('/health')
  return response.data
}

export const getCourses = async () => {
  const response = await api.get('/data/courses')
  return response.data
}

export default api